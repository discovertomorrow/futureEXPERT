from __future__ import annotations

import logging
import os
import pprint
from typing import Any, Literal, Optional, Union, cast

import pandas as pd
import pydantic

from futureexpert.batch_forecast import ReportConfig, create_forecast_payload
from futureexpert.checkin import (DataDefinition,
                                  FileSpecification,
                                  TsCreationConfig,
                                  build_payload_from_ui_config,
                                  create_checkin_payload_1,
                                  create_checkin_payload_2)
from futureexpert.future_api import FutureApiClient

pp = pprint.PrettyPrinter(indent=4)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastStatusProgress(pydantic.BaseModel):
    """Progress of a forecasting report."""
    requested: int
    pending: int
    finished: int


class ForecastStatusResults(pydantic.BaseModel):
    """Result status of a forecasting report.

    This does only include already finished runs."""
    successful: int
    no_evaluation: int
    error: int


class ForecastStatus(pydantic.BaseModel):
    """Status of a forecasting report."""
    id: ForecastIdentifier
    progress: ForecastStatusProgress
    results: ForecastStatusResults

    @property
    def is_finished(self) -> bool:
        """Indicates whether a forecasting report is finished or not."""
        return self.progress.pending == 0

    def print(self) -> None:
        """Prints a summary of the status."""
        title = f'Status forecasting report for id: {self.id}'

        if self.progress.requested == 0:
            print(f'{title}\n  No run was created')
            return

        pct_txt = f'{round(self.progress.finished/self.progress.requested*100)} % are finished'
        overall = f'{self.progress.requested} time series requested for calculation'
        finished_txt = f'{self.progress.finished} time series are finished'
        noeval_txt = f'{self.results.no_evaluation} time series are no evaluation'
        error_txt = f'{self.results.error} time series calculation run into an error'
        print(f'{title}\n {pct_txt} \n {overall} \n {finished_txt} \n {noeval_txt} \n {error_txt}')


class ForecastIdentifier(pydantic.BaseModel):
    report_id: int
    settings_id: int


class ExpertClient:
    """Future expert client."""

    def __init__(self,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 group: Optional[str] = None,
                 environment: Optional[Literal['production', 'staging', 'development']] = None) -> None:
        """Initializer.

        Parameters
        ----------
        user
            The user name for future.
            If not provided, the user is read from environment variable FUTURE_USER.
        password
            The password for future.
            If not provided, the password is read from environment variable FUTURE_PW.
        group
            Optionally the name of the future group. Only relevant if the user has access to multiple groups.
            If not provided, the group is tried to get from the environment variable FUTURE_GROUP.
        environment
            Optionally the future environment to be used, defaults to production environment.
            If not provided, the environment is tried to get from the environment variable FUTURE_ENVIRONMENT.
        """
        future_user = user or os.environ['FUTURE_USER']
        future_password = password or os.environ['FUTURE_PW']
        future_group = group or os.getenv('FUTURE_GROUP')
        future_env = cast(Literal['production', 'staging', 'development'],
                          environment or os.getenv('FUTURE_ENVIRONMENT') or 'production')

        self.client = FutureApiClient(user=future_user, password=future_password, environment=future_env)

        authorized_groups = self.client.userinfo['groups']
        if future_group is None and len(authorized_groups) == 1:
            raise ValueError(
                f'You have access to multiple groups. Please select one of the following: {authorized_groups}')
        self.switch_group(new_group=future_group or authorized_groups[0],
                          verbose=future_group is not None)

    def switch_group(self, new_group: str, verbose: bool = True) -> None:
        """Switches the current group.

        Parameters
        ----------
        new_group
            The name of the group to activate.
        verbose
            If enabled, shows the group name in the log message.
        """
        if new_group not in self.client.userinfo['groups']:
            raise RuntimeError(f'You are not authorized to access group {new_group}')
        self.group = new_group
        verbose_text = f' for group {self.group}' if verbose else ''
        logger.info(f'Successfully logged in{verbose_text}.')

    def upload_data(self, source: Union[pd.DataFrame, str]) -> Any:
        """Uploads your raw data for further processing.

        Parameters
        ----------
        source
            path to a csv file or a pandas data frame

        Returns
        -------
        Identifier for the user Inputs.
        """
        df_file = None
        if isinstance(source, pd.DataFrame):
            file_specs = FileSpecification()
            csv = source.to_csv(index=False, sep=file_specs.delimiter,
                                decimal=file_specs.decimal, encoding='utf-8-sig')
            df_file = (f'{self.group}_from_df.csv', csv)
            path = None
        else:
            path = source

        # TODO: currently only one file is supported here.
        upload_feedback = self.client.upload_user_inputs_for_group(
            self.group, path, df_file)

        return upload_feedback

    def check_data_definition(self, user_input_id: str, file_uuid: str, data_definition: DataDefinition, file_specification: FileSpecification = FileSpecification()) -> Any:
        """Checks the data definition.

        Removes specified rows and columns. Checks if column values have any issues.

        Parameters
        ----------
        user_input_id
            UUID of the user input.
        file_uuid
            UUID of the file.
        data_definition
            Defines Columns and row and column removal.
        file_specification
            needed if a csv is used with e.g. german format.
        """
        payload = create_checkin_payload_1(
            user_input_id, file_uuid, data_definition, file_specification)

        self.payload_checkin = payload

        logger.info('Started data definition using futureCHECK-IN...')
        result = self.client.execute_action(group_id=self.group,
                                            core_id='checkin-preprocessing',
                                            payload=payload,
                                            interval_status_check_in_seconds=2)

        error_message = result['error']
        if error_message != '':
            raise Exception(f'Error during the execution of the futureCHECK-IN: {error_message}')

        logger.info('Finished data definition.')
        return result

    def create_time_series(self, config_obj: Optional[TsCreationConfig] = None, json_config_file: Optional[str] = None) -> Any:
        """Second step of the checkin process which creates the time series.

        Aggregates the data and safes them to the database.

        Parameters
        ----------
        config_obj
            Configuration for the time series creation.
        json_config_file
            Path to the json file with the checkin configuration.
        """
        logger.info('Transforming input data...')

        if config_obj is None and json_config_file is None:
            raise ValueError('No configuration source is provided.')

        if config_obj is not None and json_config_file is not None:
            raise ValueError('Only one configuration source can be processed.')

        if config_obj is not None:
            payload = create_checkin_payload_2(self.payload_checkin, config_obj)
        if json_config_file is not None:
            payload = build_payload_from_ui_config(
                user_input_id=self.payload_checkin['userInputId'], file_uuid=self.payload_checkin['payload']['fileUuid'], path=json_config_file)

        logger.info('Creating time series using futureCHECK-IN...')
        result = self.client.execute_action(group_id=self.group,
                                            core_id='checkin-preprocessing',
                                            payload=payload,
                                            interval_status_check_in_seconds=2)
        error_message = result['error']
        if error_message != '':
            raise Exception(f'Error during the execution of the futureCHECK-IN: {error_message}')

        logger.info('Finished time series creation.')

        return result

    def checkin_time_series(self,
                            raw_data_source: Union[pd.DataFrame, str],
                            data_definition: Optional[DataDefinition] = None,
                            config_ts_creation: Optional[TsCreationConfig] = None,
                            config_from_json: Optional[str] = None,
                            file_specification: FileSpecification = FileSpecification()) -> Any:
        """Checks in time series data that can be used as actuals or covariate data.

        Parameters
        ----------
        raw_data_source
            data frame that contains the raw data or  path to where the csv file with the data is stored.           
        data_definition
            Defines Column and row and column removal.
        config_ts_creation
            Defines filter and aggreagtion level of the time series.
        config_from_json
            Path to the json file with the checkin configuration.       
        file_specification
            needed if a csv is used with e.g. german format.

        Returns
        -------
        Id of the time series version. Used to identifiy the time series.
        """
        upload_feedback = self.upload_data(source=raw_data_source)

        user_input_id = upload_feedback['uuid']
        file_id = upload_feedback['files'][0]['uuid']

        if data_definition is not None:
            self.check_data_definition(user_input_id=user_input_id,
                                       file_uuid=file_id,
                                       data_definition=data_definition,
                                       file_specification=file_specification)
        else:
            self.payload_checkin = {'userInputId': user_input_id,
                                    'payload': {'fileUuid': file_id}}

        response = self.create_time_series(config_obj=config_ts_creation,
                                           json_config_file=config_from_json)

        if len(response['result']['timeSeries']) > 7:
            logger.warning(
                "More than seven time series created. If these time series should be used as covariates, only the first seven will be used.")

        return response['result']['tsVersion']['_id']

    def start_forecast(self, version: str, config: ReportConfig) -> ForecastIdentifier:
        """Starts a forecasting report.

        Parameters
        ----------
        version
            ID of a time series version
        config
            Configuration of the forecasting report.

        Returns
        -------
        The identifier of the forecasting report.
        """
        logger.info('Preparing data for forecast...')
        payload = create_forecast_payload(version, config)
        logger.info('Finished data preparation for forecast.')
        logger.info('Started creating forecasting report with futureFORECAST...')
        result = self.client.execute_action(group_id=self.group,
                                            core_id='forecast-batch',
                                            payload=payload,
                                            interval_status_check_in_seconds=2)
        logger.info('Finished report creation. Forecasts are running...')
        return ForecastIdentifier.model_validate(result)

    def get_forecast_status(self, id: Union[ForecastIdentifier, int]) -> ForecastStatus:
        """Gets the current status of the forecast run.

        Parameters
        ----------
        id
            Forecast identifier or plain report ID.
        """
        report_id = id.report_id if isinstance(id, ForecastIdentifier) else id
        report_status = self.client.get_report_status(group_id=self.group, report_id=report_id)['status_summary']

        created = report_status.get('Created', 0)
        successful = report_status.get('Successful', 0)
        noeval = report_status.get('NoEvaluation', 0)
        error = report_status.get('Error', 0)
        summary = ForecastStatusProgress(requested=created,
                                         pending=created - successful - noeval - error,
                                         finished=successful + noeval + error)
        results = ForecastStatusResults(successful=successful,
                                        no_evaluation=noeval,
                                        error=error)
        return ForecastStatus(id=id, progress=summary, results=results)

    def get_results(self, id: Union[ForecastIdentifier, int], forecast_and_actuals: bool = True, backtesting: bool = False, preprocessing: bool = False) -> Any:
        """Gets the results from the given report.

        Parameters
        ----------
        id
            Forecast identifier or plain report ID.
        forecast_and_actuals
            should forecast results be returned or not.
        backtesting
            should backtesting results be returned or not.
        preprocessing
            should preprocessing results be returned or not.
        """
        fc = bt = pp = None
        report_id = id.report_id if isinstance(id, ForecastIdentifier) else id

        if forecast_and_actuals:
            fc = self.client.get_forecasts_and_actuals(group_id=self.group, report_id=report_id)

        if backtesting:
            bt = self.client.get_backtesting_results(group_id=self.group, report_id=report_id)

        if preprocessing:
            pp = self.client.get_preprocessing_results(group_id=self.group, report_id=report_id)

        return {'forecast_and_actuals': fc,
                'backtesting': bt,
                'preprocessing': pp}

    def create_forecast_from_raw_data(self,
                                      raw_data_source: Union[pd.DataFrame, str],
                                      config_fc: ReportConfig,
                                      data_definition: Optional[DataDefinition] = None,
                                      config_ts_creation: Optional[TsCreationConfig] = None,
                                      config_from_json: Optional[str] = None,
                                      file_specification: FileSpecification = FileSpecification()) -> ForecastIdentifier:
        """Starts a forecast run from raw data without the possibility to inspect interim results from the data preparation.

        Parameters
        ----------
        raw_data_source
            data frame that contains the raw data or path to where the csv file with the data is stored.        
        config_fc
            Configuration of the forecast run.
        config_data_def
            Defines Column and row and column removal.
        config_ts_creation
            Defines filter and aggreagtion level of the time series.
        config_from_json
            Path to the json file with the checkin configuration.
        file_specification
            needed if a csv is used with e.g. german format.

        Returns
        -------
        The identifier of the forecasting report.
        """
        upload_feedback = self.upload_data(source=raw_data_source)

        user_input_id = upload_feedback['uuid']
        file_id = upload_feedback['files'][0]['uuid']

        if data_definition is not None:
            self.check_data_definition(user_input_id=user_input_id,
                                       file_uuid=file_id,
                                       data_definition=data_definition,
                                       file_specification=file_specification)
        else:
            self.payload_checkin = {'userInputId': user_input_id,
                                    'payload': {'fileUuid': file_id}}

        res2 = self.create_time_series(config_obj=config_ts_creation,
                                       json_config_file=config_from_json)

        version = res2['result']['tsVersion']['_id']
        return self.start_forecast(version=version, config=config_fc)
