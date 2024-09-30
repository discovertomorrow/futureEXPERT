"""Client for connecting with future."""
from __future__ import annotations

import json
import logging
import os
import pprint
from datetime import datetime
from typing import Any, Literal, Optional, Union, cast

import dotenv
import pandas as pd
import pydantic

from futureexpert._future_api import FutureApiClient
from futureexpert._helpers import calculate_max_ts_len, snake_to_camel
from futureexpert.checkin import CheckInResult, DataDefinition, FileSpecification, TsCreationConfig
from futureexpert.forecast import ForecastResult, ReportConfig
from futureexpert.matcher import MatcherConfig, MatcherResult
from futureexpert.pool import CheckInPoolResult, PoolCovDefinition, PoolCovOverview
from futureexpert.shared_models import TimeSeries

pp = pprint.PrettyPrinter(indent=4)
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
        noeval_txt = f'{self.results.no_evaluation} time series without evaluation'
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

    @staticmethod
    def from_dotenv() -> ExpertClient:
        """Create an instance from a .env file or environment variables."""
        dotenv.load_dotenv()
        return ExpertClient()

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

    def check_data_definition(self,
                              user_input_id: str,
                              file_uuid: str,
                              data_definition: DataDefinition,
                              file_specification: FileSpecification = FileSpecification()) -> Any:
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
        payload = self._create_checkin_payload_1(
            user_input_id, file_uuid, data_definition, file_specification)

        logger.info('Started data definition using futureCHECK-IN...')
        result = self.client.execute_action(group_id=self.group,
                                            core_id='checkin-preprocessing',
                                            payload=payload,
                                            interval_status_check_in_seconds=2)

        error_message = result['error']
        if error_message != '':
            raise RuntimeError(f'Error during the execution of the futureCHECK-IN: {error_message}')

        logger.info('Finished data definition.')
        return result

    def create_time_series(self,
                           user_input_id: str,
                           file_uuid: str,
                           data_definition: Optional[DataDefinition] = None,
                           config_ts_creation: Optional[TsCreationConfig] = None,
                           config_checkin: Optional[str] = None,
                           file_specification: FileSpecification = FileSpecification()) -> Any:
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
            payload_1 = self._create_checkin_payload_1(
                user_input_id, file_uuid, data_definition, file_specification)
            payload = self._create_checkin_payload_2(payload_1, config_ts_creation)
        if config_checkin is not None:
            payload = self._build_payload_from_ui_config(
                user_input_id=user_input_id, file_uuid=file_uuid, path=config_checkin)

        logger.info('Creating time series using futureCHECK-IN...')
        result = self.client.execute_action(group_id=self.group,
                                            core_id='checkin-preprocessing',
                                            payload=payload,
                                            interval_status_check_in_seconds=2)
        error_message = result['error']
        if error_message != '':
            raise RuntimeError(f'Error during the execution of the futureCHECK-IN: {error_message}')

        logger.info('Finished time series creation.')

        return result

    def check_in_pool_covs(self,
                           requested_pool_covs: list[PoolCovDefinition]) -> CheckInPoolResult:
        """Create a new version from a list of pool covariates and version ids.

        Parameters
        ----------
        requested_pool_covs
            List of pool covariate definitions. Each definition consists of an pool_cov_id and an optional version_id.
            If no version id is provided, the newest version of the covariate is used.

        Returns
        -------
        Result object with fields version_id and pool_cov_information.
        """
        logger.info('Transforming input data...')

        payload = {
            'payload': {
                'requested_indicators': [
                    {**covariate.model_dump(exclude_none=True),
                     'indicator_id': covariate.pool_cov_id}
                    for covariate in requested_pool_covs
                ]
            }
        }
        for covariate in payload['payload']['requested_indicators']:
            covariate.pop('pool_cov_id', None)

        logger.info('Creating time series using checkin-pool...')
        result = self.client.execute_action(group_id=self.group,
                                            core_id='checkin-pool',
                                            payload=payload,
                                            interval_status_check_in_seconds=2)

        logger.info('Finished time series creation.')

        return CheckInPoolResult(**result['result'])

    def get_pool_cov_overview(self,
                              granularity: Optional[str] = None,
                              search: Optional[str] = None) -> PoolCovOverview:
        """Gets an overview of all available covariates on the futurePOOL based on the defined filters.

        Parameters
        ----------
        granularity
            If set, returns only data matching that granularity (Day or Month).
        search
            If set, performs a full-text search and only returns data found in that search.

        Returns
        -------
        PoolCovOverview object with tables containing the covariates with
        different levels of detail .
        """
        response_json = self.client.get_pool_cov_overview(granularity=granularity, search=search)
        return PoolCovOverview(response_json)

    def check_in_time_series(self,
                             raw_data_source: Union[pd.DataFrame, str],
                             data_definition: Optional[DataDefinition] = None,
                             config_ts_creation: Optional[TsCreationConfig] = None,
                             config_checkin: Optional[str] = None,
                             file_specification: FileSpecification = FileSpecification()) -> CheckInResult:
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
        Id of the time series version. Used to identifiy the time series and the values of the time series.
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

        result = [TimeSeries(**ts) for ts in response['result']['timeSeries']]
        return CheckInResult(time_series=result,
                             version_id=response['result']['tsVersion']['_id'])

    def _create_checkin_payload_1(self, user_input_id: str,
                                  file_uuid: str,
                                  data_definition: DataDefinition,
                                  file_specification: FileSpecification = FileSpecification()) -> Any:
        """Creates the payload for the futureCHECK-IN stage prepareDataset.

        Parameters
        ----------
        user_input_id
            UUID of the user input.
        file_uuid
            UUID of the file.
        data_definition
            Specifies the data, value and group columns and which rows and columns are to be removed first.
        file_specification
            Specify the format of the CSV file. Only relevant if a CSV was given as input.
        """

        return {'userInputId': user_input_id,
                'payload': {
                    'stage': 'prepareDataset',
                    'fileUuid': file_uuid,
                    'meta': file_specification.model_dump(),
                    'performedTasks': {
                        'removedRows': data_definition.remove_rows,
                        'removedCols': data_definition.remove_columns
                    },
                    'columnDefinition': {
                        'dateColumns': [{snake_to_camel(key): value for key, value in
                                        data_definition.date_columns.model_dump(exclude_none=True).items()}],
                        'valueColumns': [{snake_to_camel(key): value for key, value in d.model_dump(exclude_none=True).items()}
                                         for d in data_definition.value_columns],
                        'groupColumns': [{snake_to_camel(key): value for key, value in d.model_dump(exclude_none=True).items()}
                                         for d in data_definition.group_columns]
                    }
                }}

    def _build_payload_from_ui_config(self, user_input_id: str, file_uuid: str, path: str) -> Any:
        """Creates the payload for the futureCHECK-IN stage createDataset.

        Parameters
        ----------
        user_input_id
            UUID of the user input.
        file_uuid
            UUID of the file.
        path
            Path to the JSON file.
        """

        with open(path) as file:
            file_data = file.read()
            json_data = json.loads(file_data)

        json_data['stage'] = 'createDataset'
        json_data['fileUuid'] = file_uuid
        del json_data["performedTasksLog"]

        return {'userInputId': user_input_id,
                'payload': json_data}

    def _create_checkin_payload_2(self, payload: dict[str, Any], config: TsCreationConfig) -> Any:
        """Creates the payload for the futureCHECK-IN stage createDataset.

        Parameters
        ----------
        payload
            Payload used in `create_checkin_payload_1`.
        config
            Configuration for time series creation.
        """

        payload['payload']['rawDataReviewResults'] = {}
        payload['payload']['timeSeriesDatasetParameter'] = {
            'aggregation': {'operator': 'sum',
                            'option': config.missing_value_handler},
            'date': {
                'timeGranularity': config.time_granularity,
                'startDate': config.start_date,
                'endDate': config.end_date
            },
            'grouping': {
                'dataLevel': config.grouping_level,
                'filter':  [d.model_dump() for d in config.filter]
            },
            'values': [{snake_to_camel(key): value for key, value in d.model_dump().items()} for d in config.new_variables],
            'valueColumnsToSave': config.value_columns_to_save
        }
        payload['payload']['stage'] = 'createDataset'

        return payload

    def _create_forecast_payload(self, version: str, config: ReportConfig) -> Any:
        """Creates the payload for the forecast.

        Parameters
        ----------
        version
            Version of the time series that should get forecasts.
        config
            Configuration of the forecast run.
        """

        config_dict = config.model_dump()
        config_dict['actuals_version'] = version
        config_dict['report_note'] = config_dict['title']
        config_dict['cov_selection_report_id'] = config_dict['matcher_report_id']
        config_dict['forecasting']['n_ahead'] = config_dict['forecasting']['fc_horizon']
        config_dict['backtesting'] = config_dict['method_selection']

        if config.pool_covs is not None:
            covs_version = self.check_in_pool_covs(requested_pool_covs=config.pool_covs)
            config_dict['covs_version'] = covs_version.version_id
        config_dict.pop('pool_covs')

        config_dict.pop('title')
        config_dict['forecasting'].pop('fc_horizon')
        config_dict.pop('matcher_report_id')
        config_dict.pop('method_selection')

        payload = {'payload': config_dict}

        return payload

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
        payload = self._create_forecast_payload(version, config)
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

        return ReportStatus(id=fc_identifier,
                            progress=summary,
                            results=results,
                            error_reasons=raw_result.get('customer_specific', {}).get('log_messages', None))

    def get_fc_results(self,
                       id: Union[ReportIdentifier, int],
                       include_k_best_models: int = 1,
                       include_backtesting: bool = False) -> list[ForecastResult]:
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

    def get_matcher_results(self, id: Union[ReportIdentifier, int]) -> list[MatcherResult]:
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

    def start_forecast_from_raw_data(self,
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

        if not self.is_analyst and config.db_name is not None:
            raise ValueError('Only users with the role analyst are allowed to use the parameter db_name.')

        payload = self._create_matcher_payload(config)

        result = self.client.execute_action(group_id=self.group,
                                            core_id='cov-selection',
                                            payload=payload,
                                            interval_status_check_in_seconds=2)
        logger.info('Finished report creation.')
        return ReportIdentifier.model_validate(result)

    def _create_matcher_payload(self, config: MatcherConfig) -> Any:
        """Converts the MatcherConfig into the payload needed for the cov-selection core."""

        if config.pool_covs is not None:
            covs_version = self.check_in_pool_covs(requested_pool_covs=config.pool_covs)
            config.covs_version = covs_version.version_id

        config_dict: dict[str, Any] = {
            'report_description': config.title,
            'db_name': config.db_name,
            'data_config': {
                'actuals_version': config.actuals_version,
                'actuals_filter': config.actuals_filter,
                'covs_version': config.covs_version,
                'covs_filter': config.covs_filter,
            },
            "compute_config": {
                "evaluation_start_date": config.evaluation_start_date,
                "evaluation_end_date": config.evaluation_end_date,
                "base_report_id": None,
                "base_report_requested_run_status": None,
                "report_update_strategy": "KEEP_OWN_RUNS",
                "cov_names": {
                    'cov_name_prefix': '',
                    'cov_name_field': 'name',
                    'cov_name_suffix': '',
                },
                "preselection": {
                    "min_num_actuals_obs": 78,
                    "num_obs_short_term_class": 36,
                    "max_publication_lag": config.max_publication_lag,
                    "min_num_cov_obs": 96
                },
                "postselection": {
                    "num_obs_short_term_correlation": 60,
                    "clustering_run_id": None,
                    "post_selection_queries": config.post_selection_queries,
                    "post_selection_concatenation_operator": "&",
                    "protected_selections_queries": [],
                    "protected_selections_concatenation_operator": "&"
                },
                "lighthouse_config": {
                    "enable_leading_covariate_selection": config.enable_leading_covariate_selection,
                    "lag_selection_fixed_season_length": config.lag_selection_fixed_season_length,
                    "lag_selection_fixed_lags": config.lag_selection_fixed_lags,
                    "lag_selection_min_lag": config.lag_selection_min_lag,
                    "lag_selection_max_lag": config.lag_selection_max_lag
                }
            }
        }

        return {'payload': config_dict}
