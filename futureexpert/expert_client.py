from __future__ import annotations

import logging
import os
import pprint
from datetime import datetime
from typing import Any, Literal, Optional, Union, cast

import pandas as pd
import pydantic

from futureexpert.batch_forecast import (MatcherConfig,
                                         ReportConfig,
                                         calculate_max_ts_len,
                                         create_forecast_payload,
                                         create_matcher_payload)
from futureexpert.checkin import (DataDefinition,
                                  FileSpecification,
                                  TsCreationConfig,
                                  build_payload_from_ui_config,
                                  create_checkin_payload_1,
                                  create_checkin_payload_2)
from futureexpert.future_api import FutureApiClient
from futureexpert.result_models import ForecastResult, MatcherResult

pp = pprint.PrettyPrinter(indent=4)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportStatusProgress(pydantic.BaseModel):
    """Progress of a forecasting report."""
    requested: int
    pending: int
    finished: int


class ReportStatusResults(pydantic.BaseModel):
    """Result status of a forecasting report.

    This only includes runs that are already finished."""
    successful: int
    no_evaluation: int
    error: int


class ReportStatus(pydantic.BaseModel):
    """Status of a forecast or matcher report."""
    id: ReportIdentifier
    progress: ReportStatusProgress
    results: ReportStatusResults
    error_reasons: Optional[Any] = None

    @property
    def is_finished(self) -> bool:
        """Indicates whether a forecasting report is finished."""
        return self.progress.pending == 0

    def print(self) -> None:
        """Prints a summary of the status."""
        title = f'Status forecasting report for id: {self.id}'

        if self.progress.requested == 0:
            print(f'{title}\n  No run was created')
            return

        pct_txt = f'{round(self.progress.finished/self.progress.requested*100)} % are finished'
        overall = f'{self.progress.requested} time series requested for calculation'
        finished_txt = f'{self.progress.finished} time series finished'
        noeval_txt = f'{self.results.no_evaluation} time series no evaluation'
        error_txt = f'{self.results.error} time series ran into an error'
        print(f'{title}\n {pct_txt} \n {overall} \n {finished_txt} \n {noeval_txt} \n {error_txt}')


class ReportIdentifier(pydantic.BaseModel):
    report_id: int
    settings_id: Optional[int]


class ExpertClient:
    """FutureEXPERT client."""

    def __init__(self,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 group: Optional[str] = None,
                 environment: Optional[Literal['production', 'staging', 'development']] = None) -> None:
        """Initializer.

        Parameters
        ----------
        user
            The username for the _future_ platform.
            If not provided, the username is read from environment variable FUTURE_USER.
        password
            The password for the _future_ platform.
            If not provided, the password is read from environment variable FUTURE_PW.
        group
            Optionally the name of the futureEXPERT group. Only relevant if the user has access to multiple groups.
            If not provided, the group is read from the environment variable FUTURE_GROUP.
        environment
            Optionally the _future_ environment to be used, defaults to production environment.
            If not provided, the environment is read from the environment variable FUTURE_ENVIRONMENT.
        """
        future_user = user or os.environ['FUTURE_USER']
        future_password = password or os.environ['FUTURE_PW']
        future_group = group or os.getenv('FUTURE_GROUP')
        future_env = cast(Literal['production', 'staging', 'development'],
                          environment or os.getenv('FUTURE_ENVIRONMENT') or 'production')

        self.client = FutureApiClient(user=future_user, password=future_password, environment=future_env)

        authorized_groups = self.client.userinfo['groups']
        if future_group is None and len(authorized_groups) != 1:
            raise ValueError(
                f'You have access to multiple groups. Please select one of the following: {authorized_groups}')
        self.switch_group(new_group=future_group or authorized_groups[0],
                          verbose=future_group is not None)
        self.is_analyst = 'analyst' in self.client.user_roles
        self.forecast_core_id = 'forecast-batch-internal' if self.is_analyst else 'forecast-batch'

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
        """Uploads the given raw data for further processing.

        Parameters
        ----------
        source
            Path to a CSV file or a pandas data frame.

        Returns
        -------
        Identifier for the user Inputs.
        """
        df_file = None
        if isinstance(source, pd.DataFrame):
            file_specs = FileSpecification()
            csv = source.to_csv(index=False, sep=file_specs.delimiter,
                                decimal=file_specs.decimal, encoding='utf-8-sig')
            time_stamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
            df_file = (f'expert-{time_stamp}.csv', csv)
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
            Specifies the data, value and group columns and which rows and columns are to be removed first.
        file_specification
            Needed if a CSV is used with e.g. German format.
        """
        payload = create_checkin_payload_1(
            user_input_id, file_uuid, data_definition, file_specification)

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

    def create_time_series(self, user_input_id: str, file_uuid: str, data_definition: Optional[DataDefinition] = None, config_ts_creation: Optional[TsCreationConfig] = None, config_checkin: Optional[str] = None, file_specification: FileSpecification = FileSpecification()) -> Any:
        """Last step of the futureCHECK-IN process which creates the time series.

        Aggregates the data and saves them to the database.

        Parameters
        ----------
        user_input_id
            UUID of the user input.
        file_uuid
            UUID of the file.
        data_definition
            Specifies the data, value and group columns and which rows and columns are to be removed first.
        file_specification
            Needed if a CSV is used with e.g. German format.
        config_ts_creation
            Configuration for the time series creation.
        config_checkin
            Path to the JSON file with the futureCHECK-IN configuration. `config_ts_creation` and `config_checkin` 
            cannot be set simultaneously. The configuration may be obtained from the last step of 
            futureCHECK-IN using the _future_ frontend (future.prognostica.de).
        """
        logger.info('Transforming input data...')

        if config_ts_creation is None and config_checkin is None:
            raise ValueError('No configuration source is provided.')

        if config_ts_creation is not None and config_checkin is not None:
            raise ValueError('Only one configuration source can be processed.')

        if config_checkin is None and (data_definition is None or config_ts_creation is None):
            raise ValueError(
                'For checkin configuration via python `data_defintion`and `config_ts_cration` must be provided.')

        if config_ts_creation is not None and data_definition is not None:
            payload_1 = create_checkin_payload_1(
                user_input_id, file_uuid, data_definition, file_specification)
            payload = create_checkin_payload_2(payload_1, config_ts_creation)
        if config_checkin is not None:
            payload = build_payload_from_ui_config(
                user_input_id=user_input_id, file_uuid=file_uuid, path=config_checkin)

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

    def check_in_time_series(self,
                             raw_data_source: Union[pd.DataFrame, str],
                             data_definition: Optional[DataDefinition] = None,
                             config_ts_creation: Optional[TsCreationConfig] = None,
                             config_checkin: Optional[str] = None,
                             file_specification: FileSpecification = FileSpecification()) -> Any:
        """Checks in time series data that can be used as actuals or covariate data.

        Parameters
        ----------
        raw_data_source
            Data frame that contains the raw data or path to where the CSV file with the data is stored. 
        data_definition
            Specifies the data, value and group columns and which rows and columns are to be removed.
        config_ts_creation
            Defines filter and aggreagtion level of the time series.
        config_checkin
            Path to the JSON file with the futureCHECK-IN configuration. `config_ts_creation` and `config_checkin` 
            cannot be set simultaneously. The configuration may be obtained from the last step of 
            futureCHECK-IN using the future frontend (future.prognostica.de).
        file_specification
            Needed if a CSV is used with e.g. German format.

        Returns
        -------
        Id of the time series version. Used to identifiy the time series.
        """
        upload_feedback = self.upload_data(source=raw_data_source)

        user_input_id = upload_feedback['uuid']
        file_id = upload_feedback['files'][0]['uuid']

        response = self.create_time_series(user_input_id=user_input_id,
                                           file_uuid=file_id,
                                           data_definition=data_definition,
                                           config_ts_creation=config_ts_creation,
                                           config_checkin=config_checkin,
                                           file_specification=file_specification)

        if len(response['result']['timeSeries']) > 7:
            logger.warning(
                "More than seven time series created. If these time series should be used as covariates, only the first seven will be used.")

        return response['result']['tsVersion']['_id']

    def start_forecast(self, version: str, config: ReportConfig) -> ReportIdentifier:
        """Starts a forecasting report.

        Parameters
        ----------
        version
            ID of a time series version.
        config
            Configuration of the forecasting report.

        Returns
        -------
        The identifier of the forecasting report.
        """

        version_data = self.client.get_ts_version(self.group, version)
        config.max_ts_len = calculate_max_ts_len(max_ts_len=config.max_ts_len,
                                                 granularity=version_data['customer_specific']['granularity'])
        logger.info('Preparing data for forecast...')

        if not self.is_analyst and (config.db_name is not None or config.priority is not None):
            raise ValueError('Only users with the role analyst are allowed to use the parameters db_name and priority.')
        payload = create_forecast_payload(version, config)
        logger.info('Finished data preparation for forecast.')
        logger.info('Started creating forecasting report with futureFORECAST...')
        result = self.client.execute_action(group_id=self.group,
                                            core_id=self.forecast_core_id,
                                            payload=payload,
                                            interval_status_check_in_seconds=2)
        logger.info('Finished report creation. Forecasts are running...')
        return ReportIdentifier.model_validate(result)

    def get_report_status(self, id: Union[ReportIdentifier, int], include_error_reason: bool = True) -> ReportStatus:
        """Gets the current status of a forecast or matcher report.

        Parameters
        ----------
        id
            Report identifier or plain report ID.
        include_error_reason
            Determines whether log messages are to be included in the result.

        """
        fc_identifier = id if isinstance(id, ReportIdentifier) else ReportIdentifier(report_id=id, settings_id=None)
        raw_result = self.client.get_report_status(
            group_id=self.group, report_id=fc_identifier.report_id, include_error_reason=include_error_reason)

        report_status = raw_result['status_summary']
        created = report_status.get('Created', 0)
        successful = report_status.get('Successful', 0)
        noeval = report_status.get('NoEvaluation', 0)
        error = report_status.get('Error', 0)
        summary = ReportStatusProgress(requested=created,
                                       pending=created - successful - noeval - error,
                                       finished=successful + noeval + error)
        results = ReportStatusResults(successful=successful,
                                      no_evaluation=noeval,
                                      error=error)

        return ReportStatus(id=fc_identifier, progress=summary, results=results, error_reasons=raw_result.get('customer_specific', {}).get('log_messages', None))

    def get_fc_results(self, id: Union[ReportIdentifier, int], include_k_best_models: int = 1, include_backtesting: bool = False) -> Any:
        """Gets the results from the given report.

        Parameters
        ----------
        id
            Forecast identifier or plain report ID.
        include_k_best_models
            Number of k best models for which results are to be returned.           
        include_backtesting
            Determines whether backtesting results are to be returned.
        """

        if include_k_best_models < 1:
            raise ValueError('At least one model is needed.')

        report_id = id.report_id if isinstance(id, ReportIdentifier) else id

        results = self.client.get_fc_results(group_id=self.group,
                                             report_id=report_id,
                                             include_k_best_models=include_k_best_models,
                                             include_backtesting=include_backtesting)

        return [ForecastResult(**result) for result in results]

    def get_matcher_results(self, id: Union[ReportIdentifier, int]) -> Any:
        """Gets the results from the given report.

        Parameters
        ----------
        id
            Report identifier or plain report ID.
        """

        report_id = id.report_id if isinstance(id, ReportIdentifier) else id

        results = self.client.get_matcher_results(group_id=self.group,
                                                  report_id=report_id)

        return [MatcherResult(**result) for result in results]

    def create_forecast_from_raw_data(self,
                                      raw_data_source: Union[pd.DataFrame, str],
                                      config_fc: ReportConfig,
                                      data_definition: Optional[DataDefinition] = None,
                                      config_ts_creation: Optional[TsCreationConfig] = None,
                                      config_checkin: Optional[str] = None,
                                      file_specification: FileSpecification = FileSpecification()) -> ReportIdentifier:
        """Starts a forecast run from raw data without the possibility to inspect interim results from the data preparation.

        Parameters
        ----------
        raw_data_source
            A Pandas DataFrame that contains the raw data or path to where the CSV file with the data is stored.   
        config_fc
            The configuration of the forecast run.
        data_definition
            Specifies the data, value and group columns and which rows and columns should be removed.
        config_ts_creation
            Defines filter and aggreagtion level of the time series.
        config_checkin
            Path to the JSON file with the futureCHECK-IN configuration. `config_ts_creation` and `config_checkin` 
            cannot be set simultaneously. The configuration may be obtained from the last step of 
            futureCHECK-IN using the future frontend (future.prognostica.de).
        file_specification
            Needed if a CSV is used with e.g. German format.

        Returns
        -------
        The identifier of the forecasting report.
        """
        upload_feedback = self.upload_data(source=raw_data_source)

        user_input_id = upload_feedback['uuid']
        file_id = upload_feedback['files'][0]['uuid']

        res2 = self.create_time_series(user_input_id=user_input_id,
                                       file_uuid=file_id,
                                       data_definition=data_definition,
                                       config_ts_creation=config_ts_creation,
                                       config_checkin=config_checkin,
                                       file_specification=file_specification)

        version = res2['result']['tsVersion']['_id']
        return self.start_forecast(version=version, config=config_fc)

    def start_matcher(self, config: MatcherConfig) -> ReportIdentifier:
        """Starts a covariate matcher report.

        Parameters
        ----------
        version
            ID of a time series version
        config
            Configuration of the covariate matcher report.

        Returns
        -------
        The identifier of the covariate matcher report.
        """

        payload = create_matcher_payload(config)

        result = self.client.execute_action(group_id=self.group,
                                            core_id='cov-selection',
                                            payload=payload,
                                            interval_status_check_in_seconds=2)
        logger.info('Finished report creation.')
        return ReportIdentifier.model_validate(result)
