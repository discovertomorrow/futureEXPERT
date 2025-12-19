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
from futureexpert.associator import AssociatorConfig, AssociatorResult
from futureexpert.checkin import CheckInResult, DataDefinition, FileSpecification, TimeSeriesVersion, TsCreationConfig
from futureexpert.forecast import ForecastResult, ForecastResults, ReportConfig
from futureexpert.forecast_consistency import (ConsistentForecastMetadata,
                                               MakeForecastConsistentConfiguration,
                                               MakeForecastConsistentDataSelection,
                                               ReconciliationConfig)
from futureexpert.matcher import MatcherConfig, MatcherResult
from futureexpert.pool import CheckInPoolResult, PoolCovDefinition, PoolCovOverview
from futureexpert.shared_models import TimeSeries

pp = pprint.PrettyPrinter(indent=4)
logger = logging.getLogger(__name__)


class MissingCredentialsError(RuntimeError):
    def __init__(self, missing_credential_type: str) -> None:
        super().__init__(f'Please enter {missing_credential_type} either when ' +
                         'initializing the expert client or in the the .env file!')


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


class ErrorReason(pydantic.BaseModel):
    """Details about a specific error in a report.

    Parameters
    ----------
    status
        The status of the run ('Error' or 'NoEvaluation').
    error_message
        The error message describing what went wrong.
    timeseries
        List of time series names that encountered this error.
    """
    status: str
    error_message: Optional[str]
    timeseries: list[str]

    @staticmethod
    def parse_error_reasons(customer_specific: dict[str, Any]) -> list[ErrorReason]:
        """Creates error reasons from raw customer_specific object."""
        log_messages = customer_specific.get('log_messages', None)
        assert log_messages is not None, 'missing log_messages property in customer_specific'
        assert isinstance(log_messages, list), 'unexpected type of log_messages'
        return [ErrorReason.model_validate(msg) for msg in log_messages]


class ReportStatus(pydantic.BaseModel):
    """Status of a forecast or matcher report.

    Parameters
    ----------
    id
        The identifier of the report.
    description
        The description of the report.
    result_type
        The result type of the report.
    progress
        Progress summary of the report.
    results
        Success/error summary of the report.
    error_reasons
        Details about the errors of the report. Each error reason contains the status,
        error message, and list of affected time series.
    prerequisites
        If the status was requested for a report that depends on other reports (ChainedReportIdentifier)
        all other report statuses are contained in the prerequisites in order to get an easy overview.
    """
    id: ReportIdentifier
    description: str
    result_type: str
    progress: ReportStatusProgress
    results: ReportStatusResults
    error_reasons: Optional[list[ErrorReason]] = None
    prerequisites: list[ReportStatus] = pydantic.Field(default_factory=list)

    @property
    def is_finished(self) -> bool:
        """Indicates whether a forecasting report is finished."""
        return self.progress.pending == 0

    def print(self, print_prerequisites: bool = True, print_error_reasons: bool = True) -> None:
        """Prints a summary of the status.

        Parameters
        ----------
        print_prerequisites
            Enable or disable printing of prerequisite reports.
        print_error_reasons
            Enable or disable printing of error reasons.
        """
        title = f'Status of report "{self.description}" of type "{self.result_type}":'
        run_description = 'time series' if self.result_type in ['forecast', 'matcher'] else 'runs'
        if print_prerequisites:
            for prerequisite in self.prerequisites:
                prerequisite.print(print_error_reasons=print_error_reasons)

        if self.progress.requested == 0:
            print(f'{title}\n  No {run_description} created')
            return

        pct_txt = f'{round(self.progress.finished/self.progress.requested*100)} % are finished'
        overall = f'{self.progress.requested} {run_description} requested for calculation'
        finished_txt = f'{self.progress.finished} {run_description} finished'
        noeval_txt = f'{self.results.no_evaluation} {run_description} without evaluation'
        error_txt = f'{self.results.error} {run_description} ran into an error'
        print(f'{title}\n {pct_txt} \n {overall} \n {finished_txt} \n {noeval_txt} \n {error_txt}')

        if print_error_reasons and self.error_reasons is not None and len(self.error_reasons) > 0:
            print('\nError reasons:')
            for error_reason in self.error_reasons:
                ts_count = len(error_reason.timeseries)
                ts_names = ', '.join(error_reason.timeseries[:3])  # Show first 3 time series
                if ts_count > 3:
                    ts_names += f' ... and {ts_count - 3} more'
                print(f'  [{error_reason.status}] {error_reason.error_message if error_reason.error_message else ""}')
                print(f'    Affected time series ({ts_count}): {ts_names}')


class ReportIdentifier(pydantic.BaseModel):
    """Report ID and Settings ID of a report. Required to identify the report, e.g. when retrieving the results."""
    report_id: int
    settings_id: Optional[int]


class ChainedReportIdentifier(ReportIdentifier):
    """Extended report identifier with prerequisites."""
    prerequisites: list[ReportIdentifier]

    @classmethod
    def of(cls, final_report_identifier: ReportIdentifier, prerequisites: list[ReportIdentifier]) -> ChainedReportIdentifier:
        return cls(report_id=final_report_identifier.report_id,
                   settings_id=final_report_identifier.settings_id,
                   prerequisites=prerequisites)


class ReportSummary(pydantic.BaseModel):
    """Report ID and description of a report."""
    report_id: int
    description: str
    result_type: str


class ExpertClient:
    """FutureEXPERT client."""

    def __init__(self,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 totp: Optional[str] = None,
                 refresh_token: Optional[str] = None,
                 group: Optional[str] = None,
                 environment: Optional[Literal['production', 'staging', 'development']] = None) -> None:
        """Initializer.

        Login using either your user credentials or a valid refresh token.

        Parameters
        ----------
        user
            The username for the _future_ platform.
            If not provided, the username is read from environment variable FUTURE_USER.
        password
            The password for the _future_ platform.
            If not provided, the password is read from environment variable FUTURE_PW.
        totp
            Optional second factor for authentication using user credentials.
        refresh_token
            Alternative login using a refresh token only instead of user credentials.
            If not provided, the token is read from the environment variable FUTURE_REFRESH_TOKEN.
            You can retrieve a long-lived refresh token (offline token) from our identity provider
            using Open ID Connect scope `offline_access` at the token endpoint. Example:
            curl -s -X POST 'https://future-auth.prognostica.de/realms/future/protocol/openid-connect/token' \
                    -H 'Content-Type: application/x-www-form-urlencoded' \
                    --data-urlencode 'client_id=expert' \
                    --data-urlencode 'grant_type=password' \
                    --data-urlencode 'scope=openid offline_access' \
                    --data-urlencode "username=$FUTURE_USER" \
                    --data-urlencode "password=$FUTURE_PW" | jq .refresh_token
        group
            Optionally the name of the futureEXPERT group. Only relevant if the user has access to multiple groups.
            If not provided, the group is read from the environment variable FUTURE_GROUP.
        environment
            Optionally the _future_ environment to be used, defaults to production environment.
            If not provided, the environment is read from the environment variable FUTURE_ENVIRONMENT.
        """
        future_env = cast(Literal['production', 'staging', 'development'],
                          environment or os.getenv('FUTURE_ENVIRONMENT') or 'production')
        future_refresh_token = refresh_token or os.getenv('FUTURE_REFRESH_TOKEN')
        if future_refresh_token:
            self.api_client = FutureApiClient(refresh_token=future_refresh_token, environment=future_env)
        else:
            try:
                future_user = user or os.environ['FUTURE_USER']
            except KeyError:
                raise MissingCredentialsError('username') from None
            try:
                future_password = password or os.environ['FUTURE_PW']
            except KeyError:
                raise MissingCredentialsError('password') from None

            self.api_client = FutureApiClient(user=future_user, password=future_password,
                                              environment=future_env, totp=totp)

        authorized_groups = self.api_client.userinfo['groups']
        future_group = group or os.getenv('FUTURE_GROUP')
        if future_group is None and len(authorized_groups) != 1:
            raise ValueError(
                f'You have access to multiple groups. Please select one of the following: {authorized_groups}')
        self.switch_group(new_group=future_group or authorized_groups[0],
                          verbose=future_group is not None)
        self.is_analyst = 'analyst' in self.api_client.user_roles
        self.forecast_core_id = 'forecast-batch-internal' if self.is_analyst else 'forecast-batch'
        self.matcher_core_id = 'cov-selection-internal' if self.is_analyst else 'cov-selection'
        self.associator_core_id = 'associator-internal' if self.is_analyst else 'associator'
        self.hcfc_core_id = 'hcfc-internal' if self.is_analyst else 'hcfc'

    def __enter__(self) -> ExpertClient:
        return self

    def __exit__(self,
                 exc_type: Optional[type[BaseException]],
                 exc_value: Optional[BaseException],
                 exc_tb: Any) -> None:
        """Cancel token refresh and logout if not a offline token is used."""
        self.api_client.auto_refresh = False
        if 'offline_access' not in self.api_client.token['scope']:
            self.logout()

    @staticmethod
    def from_dotenv() -> ExpertClient:
        """Create an instance from a .env file or environment variables."""
        dotenv.load_dotenv()
        return ExpertClient()

    def logout(self) -> None:
        """Logout from futureEXPERT.

        If logged in with a refresh token. The refresh token is revoked.
        """
        self.api_client.keycloak_openid.logout(self.api_client.token['refresh_token'])
        self.api_client.auto_refresh = False
        logger.info('Successfully logged out.')

    def switch_group(self, new_group: str, verbose: bool = True) -> None:
        """Switches the current group.

        Parameters
        ----------
        new_group
            The name of the group to activate.
        verbose
            If enabled, shows the group name in the log message.
        """
        if new_group not in self.api_client.userinfo['groups']:
            raise RuntimeError(f'You are not authorized to access group {new_group}')
        self.group = new_group
        verbose_text = f' for group {self.group}' if verbose else ''
        logger.info(f'Successfully logged in{verbose_text}.')

    def upload_data(self, source: Union[pd.DataFrame, str], file_specification: Optional[FileSpecification] = None) -> Any:
        """Uploads the given raw data for further processing.

        Parameters
        ----------
        source
            Path to a CSV file or a pandas data frame.
        file_specification
            If source is a pandas data frame, it will be uploaded as a csv using the specified parameters or the default ones.
            The parameter has no effect if source is a path to a CSV file.

        Returns
        -------
        Identifier for the user Inputs.
        """
        df_file = None
        if isinstance(source, pd.DataFrame):
            if not file_specification:
                file_specification = FileSpecification()
            csv = source.to_csv(index=False, sep=file_specification.delimiter,
                                decimal=file_specification.decimal, encoding='utf-8-sig')
            time_stamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
            df_file = (f'expert-{time_stamp}.csv', csv)
            path = None
        else:
            path = source

        # TODO: currently only one file is supported here.
        upload_feedback = self.api_client.upload_user_inputs_for_group(self.group, path, df_file)

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

        logger.info('Started data definition using CHECK-IN...')
        result = self.api_client.execute_action(group_id=self.group,
                                                core_id='checkin-preprocessing',
                                                payload=payload,
                                                interval_status_check_in_seconds=5)

        error_message = result['error']
        if error_message != '':
            raise RuntimeError(f'Error during the execution of CHECK-IN: {error_message}')

        logger.info('Finished data definition.')
        return result

    def create_time_series(self,
                           user_input_id: str,
                           file_uuid: str,
                           data_definition: Optional[DataDefinition] = None,
                           config_ts_creation: Optional[TsCreationConfig] = None,
                           config_checkin: Optional[str] = None,
                           file_specification: FileSpecification = FileSpecification()) -> Any:
        """Last step of the CHECK-IN process which creates the time series.

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
            Path to the JSON file with the CHECK-IN configuration. `config_ts_creation` and `config_checkin`
            cannot be set simultaneously. The configuration may be obtained from the last step of
            CHECK-IN using the _future_ frontend (now.future-forecasting.de).
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

        logger.info('Creating time series using CHECK-IN...')
        result = self.api_client.execute_action(group_id=self.group,
                                                core_id='checkin-preprocessing',
                                                payload=payload,
                                                interval_status_check_in_seconds=5)
        error_message = result['error']
        if error_message != '':
            raise RuntimeError(f'Error during the execution of CHECK-IN: {error_message}')

        logger.info('Finished time series creation.')

        return result

    def check_in_pool_covs(self,
                           requested_pool_covs: list[PoolCovDefinition],
                           description: Optional[str] = None) -> CheckInPoolResult:
        """Create a new version from a list of pool covariates and version ids.

        Parameters
        ----------
        requested_pool_covs
            List of pool covariate definitions. Each definition consists of an pool_cov_id and an optional version_id.
            If no version id is provided, the newest version of the covariate is used.
        description
            A short description of the selected covariates.

        Returns
        -------
        Result object with fields version_id and pool_cov_information.
        """
        logger.info('Transforming input data...')

        payload: dict[str, Any] = {
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

        payload['payload']['version_description'] = description

        logger.info('Creating time series using checkin-pool...')
        result = self.api_client.execute_action(group_id=self.group,
                                                core_id='checkin-pool',
                                                payload=payload,
                                                interval_status_check_in_seconds=5)

        logger.info('Finished time series creation.')

        return CheckInPoolResult(**result['result'])

    def get_pool_cov_overview(self,
                              granularity: Optional[str] = None,
                              search: Optional[str] = None) -> PoolCovOverview:
        """Gets an overview of all covariates available on POOL according to the given filters.

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
        response_json = self.api_client.get_pool_cov_overview(granularity=granularity, search=search)
        return PoolCovOverview(response_json)

    def get_time_series(self,
                        version_id: str) -> CheckInResult:
        """Get time series data. From previously checked-in data.

        Parameters
        ---------
        version_id
            Id of the time series version.
        Returns
        -------
        Id of the time series version. Used to identifiy the time series and the values of the time series.
        """
        result = self.api_client.get_ts_data(self.group, version_id)
        return CheckInResult(time_series=[TimeSeries(**ts) for ts in result],
                             version_id=version_id)

    def check_in_time_series(self,
                             raw_data_source: Union[pd.DataFrame, str],
                             data_definition: Optional[DataDefinition] = None,
                             config_ts_creation: Optional[TsCreationConfig] = None,
                             config_checkin: Optional[str] = None,
                             file_specification: FileSpecification = FileSpecification()) -> str:
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
            Path to the JSON file with the CHECK-IN configuration. `config_ts_creation` and `config_checkin`
            cannot be set simultaneously. The configuration may be obtained from the last step of
            CHECK-IN using the future frontend (now.future-forecasting.de).
        file_specification
            Needed if a CSV is used with e.g. German format.

        Returns
        -------
        Id of the time series version. Used to identifiy the time series.
        """
        upload_feedback = self.upload_data(source=raw_data_source, file_specification=file_specification)

        user_input_id = upload_feedback['uuid']
        file_id = upload_feedback['files'][0]['uuid']

        response = self.create_time_series(user_input_id=user_input_id,
                                           file_uuid=file_id,
                                           data_definition=data_definition,
                                           config_ts_creation=config_ts_creation,
                                           config_checkin=config_checkin,
                                           file_specification=file_specification)

        return str(response['result']['tsVersion'])

    def _create_checkin_payload_1(self, user_input_id: str,
                                  file_uuid: str,
                                  data_definition: DataDefinition,
                                  file_specification: FileSpecification = FileSpecification()) -> Any:
        """Creates the payload for the CHECK-IN stage prepareDataset.

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
                                        data_definition.date_column.model_dump(exclude_none=True).items()}],
                        'valueColumns': [{snake_to_camel(key): value for key, value in d.model_dump(exclude_none=True).items()}
                                         for d in data_definition.value_columns],
                        'groupColumns': [{snake_to_camel(key): value for key, value in d.model_dump(exclude_none=True).items()}
                                         for d in data_definition.group_columns]
                    }
                }}

    def _build_payload_from_ui_config(self, user_input_id: str, file_uuid: str, path: str) -> Any:
        """Creates the payload for the CHECK-IN stage createDataset.

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
        """Creates the payload for the CHECK-IN stage createDataset.

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
                'saveHierarchy': config.save_hierarchy,
                'filter':  [d.model_dump() for d in config.filter]
            },
            'values': [{snake_to_camel(key): value for key, value in d.model_dump().items()} for d in config.new_variables],
            'valueColumnsToSave': config.value_columns_to_save
        }
        payload['payload']['versionDescription'] = config.description
        payload['payload']['stage'] = 'createDataset'

        return payload

    def _create_reconciliation_payload(self, config: MakeForecastConsistentConfiguration) -> Any:
        """Creates the payload for forecast reconciliation.

        Parameters
        ----------
        config
            Configuration of the make forecast consistent run.
        """
        config_dict = config.model_dump()
        return {'payload': config_dict}

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

        if config.rerun_report_id:
            config_dict['base_report_id'] = config.rerun_report_id
            config_dict['report_update_strategy'] = 'KEEP_OWN_RUNS'

            base_report_requested_run_status = ['Successful']
            if 'NoEvaluation' not in config.rerun_status:
                base_report_requested_run_status.append('NoEvaluation')
            config_dict['base_report_requested_run_status'] = base_report_requested_run_status

        if config.pool_covs is not None:
            pool_covs_checkin_result = self.check_in_pool_covs(requested_pool_covs=config.pool_covs)
            cast(list[str], config_dict['covs_versions']).append(pool_covs_checkin_result.version_id)
        config_dict.pop('pool_covs')

        config_dict.pop('title')
        config_dict['forecasting'].pop('fc_horizon')
        config_dict.pop('matcher_report_id')
        config_dict.pop('method_selection')
        config_dict.pop('rerun_report_id')
        config_dict.pop('rerun_status')

        payload = {'payload': config_dict}

        return payload

    def start_associator(self, config: AssociatorConfig) -> ReportIdentifier:
        """Sarts an associator report.

        Parameters
        ----------
        config
            Configuration of the associator run.

        Returns
        -------
        The identifier of the associator report.
        """

        config_dict = config.model_dump()
        payload = {'payload': config_dict}

        result = self.api_client.execute_action(group_id=self.group,
                                                core_id=self.associator_core_id,
                                                payload=payload,
                                                interval_status_check_in_seconds=5,
                                                check_intermediate_result=True)

        report = ReportIdentifier.model_validate(result)
        logger.info(f'Report created with ID {report.report_id}. Associator is running...')
        return report

    def start_forecast(self,
                       version: str,
                       config: ReportConfig,
                       reconciliation_config: Optional[ReconciliationConfig] = None) -> ReportIdentifier:
        """Starts a forecasting report.

        Parameters
        ----------
        version
            ID of a time series version.
        config
            Configuration of the forecasting report.
        reconciliation_config
            Configuration to make forecasts consistent over hierarchical levels.
            Reconciliation assumes time series are measured in comparable units.

        Returns
        -------
        The identifier of the forecasting report.
        """
        if not self.is_analyst and (config.db_name is not None or config.priority is not None):
            raise ValueError('Only users with the role analyst are allowed to use the parameters db_name and priority.')
        if reconciliation_config is not None and reconciliation_config.enforce_forecast_minimum_constraint:
            raise ValueError('Minimum constraints for forecasts are only available via start_making_forecast_consistent.')
        version_data = self.api_client.get_ts_version(self.group, version)
        config.max_ts_len = calculate_max_ts_len(max_ts_len=config.max_ts_len,
                                                 granularity=version_data['customer_specific']['granularity'])

        logger.info('Preparing data for forecast...')
        payload = self._create_forecast_payload(version, config)
        logger.info('Finished data preparation for forecast.')

        logger.info('Started creating forecasting report with FORECAST...')
        result = self.api_client.execute_action(group_id=self.group,
                                                core_id=self.forecast_core_id,
                                                payload=payload,
                                                interval_status_check_in_seconds=5)

        forecast_identifier = ReportIdentifier.model_validate(result)
        logger.info(f'Report created with ID {forecast_identifier.report_id}. Forecasts are running...')

        if reconciliation_config is None:
            return forecast_identifier

        # Continue with forecast reconciliation
        data_selection = MakeForecastConsistentDataSelection(
            version=version, fc_report_id=forecast_identifier.report_id)
        forecast_consistency_config = MakeForecastConsistentConfiguration(db_name=config.db_name,
                                                                          reconciliation=reconciliation_config,
                                                                          data_selection=data_selection,
                                                                          report_note=config.title)
        forecast_consistency_identifier = self.start_making_forecast_consistent(config=forecast_consistency_config)
        return ChainedReportIdentifier.of(final_report_identifier=forecast_consistency_identifier,
                                          prerequisites=[forecast_identifier])

    def start_making_forecast_consistent(self, config: MakeForecastConsistentConfiguration) -> ReportIdentifier:
        """Starts process of making forecasts hierarchically consistent.

        Parameters
        ----------
        config
            Configuration of the make forecast consistent run.

        Returns
        -------
        The identifier of the forecasting report.
        """

        logger.info('Preparing data for forecast consistency...')
        if not self.is_analyst and (config.db_name is not None):
            raise ValueError('Only users with the role analyst are allowed to use the parameters db_name.')
        payload = self._create_reconciliation_payload(config)
        logger.info('Finished data preparation for forecast consistency.')

        logger.info('Started creating hierarchical reconciliation for consistent forecasts...')
        result = self.api_client.execute_action(group_id=self.group,
                                                core_id=self.hcfc_core_id,
                                                payload=payload,
                                                interval_status_check_in_seconds=5,
                                                check_intermediate_result=True)

        report = ReportIdentifier.model_validate(result)
        logger.info(f'Report created with ID {report.report_id}. Reconciliation is running...')
        return report

    def get_report_type(self, report_identifier: Union[int, ReportIdentifier]) -> str:
        """Gets the available reports, ordered from newest to oldest.

        Parameters
        ----------
        skip
            The number of initial elements of the report list to skip
        limit
            The limit on the length of the report list

        Returns
        -------
            String representation of the type of one report.
        """
        report_id = report_identifier.report_id if isinstance(
            report_identifier, ReportIdentifier) else report_identifier
        return self.api_client.get_report_type(group_id=self.group, report_id=report_id)

    def get_reports(self, skip: int = 0, limit: int = 100) -> pd.DataFrame:
        """Gets the available reports, ordered from newest to oldest.

        Parameters
        ----------
        skip
            The number of initial elements of the report list to skip
        limit
            The limit on the length of the report list

        Returns
        -------
        The available reports from newest to oldest.
        """
        group_reports = self.api_client.get_group_reports(group_id=self.group, skip=skip, limit=limit)
        vallidated_report_summarys = [ReportSummary.model_validate(report) for report in group_reports]
        return pd.DataFrame([report_summary.model_dump() for report_summary in vallidated_report_summarys])

    def _get_single_report_status(self, report_identifier: ReportIdentifier, include_error_reason: bool = True) -> ReportStatus:
        """Gets the current status of a single report.

        Parameters
        ----------
        id
            Report identifier.
        include_error_reason
            Determines whether log messages are to be included in the result.

        Returns
        -------
        The status of the report.
        """
        raw_result = self.api_client.get_report_status(group_id=self.group,
                                                       report_id=report_identifier.report_id,
                                                       include_error_reason=include_error_reason)

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
        customer_specific = raw_result.get('customer_specific', None)
        assert (customer_specific is None
                or isinstance(customer_specific, dict)), 'unexpected type of customer_specific property'

        if include_error_reason:
            assert customer_specific is not None, 'missing customer_specific property in report status'
            error_reasons = ErrorReason.parse_error_reasons(customer_specific)
        else:
            error_reasons = None

        return ReportStatus(id=report_identifier,
                            description=raw_result['description'],
                            result_type=raw_result['result_type'],
                            progress=summary,
                            results=results,
                            error_reasons=error_reasons)

    def get_report_status(self, id: Union[ReportIdentifier, int], include_error_reason: bool = True) -> ReportStatus:
        """Gets the current status of a report.

        If the provided report identifier includes prerequisites, the status of the prerequisites is included, too.

        Parameters
        ----------
        id
            Report identifier or plain report ID.
        include_error_reason
            Determines whether log messages are to be included in the result.

        Returns
        -------
        The status of the report.
        """
        identifier = id if isinstance(id, ReportIdentifier) else ReportIdentifier(report_id=id, settings_id=None)
        final_status = self._get_single_report_status(
            report_identifier=identifier, include_error_reason=include_error_reason)
        if isinstance(identifier, ChainedReportIdentifier):
            for prerequisite_identifier in identifier.prerequisites:
                prerequisite_status = self.get_report_status(id=prerequisite_identifier,
                                                             include_error_reason=include_error_reason)
                final_status.prerequisites.append(prerequisite_status)
        return final_status

    def get_fc_results(self,
                       id: Union[ReportIdentifier, int],
                       include_k_best_models: int = 1,
                       include_backtesting: bool = False,
                       include_discarded_models: bool = False) -> ForecastResults:
        """Gets the results from the given report.

        Parameters
        ----------
        id
            Forecast identifier or plain report ID.
        include_k_best_models
            Number of k best models for which results are to be returned.
        include_backtesting
            Determines whether backtesting results are to be returned.
        include_discarded_models
            Determines if models excluded from ranking should be included in the result.
        """
        report_id = id.report_id if isinstance(id, ReportIdentifier) else id
        if self.get_report_type(report_identifier=report_id) not in ['forecast',
                                                                     'MongoForecastingResultSink',
                                                                     'hierarchical-forecast']:
            raise ValueError('The given report ID does not belong to a FORECAST result. ' +
                             'Please input a different ID or use another result getter function.')
        if include_k_best_models < 1:
            raise ValueError('At least one model is needed.')

        raw_forecast_results = self.api_client.get_fc_results(group_id=self.group,
                                                              report_id=report_id,
                                                              include_k_best_models=include_k_best_models,
                                                              include_backtesting=include_backtesting,
                                                              include_discarded_models=include_discarded_models)
        result = ForecastResults(forecast_results=[ForecastResult.model_validate(result)
                                 for result in raw_forecast_results['forecast_results']])

        raw_forecast_consistency = raw_forecast_results['consistency']
        if raw_forecast_consistency is not None:
            result.consistency = ConsistentForecastMetadata.model_validate(raw_forecast_consistency)

        return result

    def get_matcher_results(self, id: Union[ReportIdentifier, int]) -> list[MatcherResult]:
        """Gets the results from the given report.

        Parameters
        ----------
        id
            Report identifier or plain report ID.
        """

        if self.get_report_type(report_identifier=id) not in ['matcher', 'CovariateSelection']:
            raise ValueError('The given report ID does not belong to a MATCHER result. ' +
                             'Please input a different ID or use another result getter function.')

        report_id = id.report_id if isinstance(id, ReportIdentifier) else id

        results = self.api_client.get_matcher_results(group_id=self.group,
                                                      report_id=report_id)

        return [MatcherResult(**result) for result in results]

    def get_associator_results(self, id: Union[ReportIdentifier, int]) -> AssociatorResult:
        """Gets the results from the given report.

        Parameters
        ----------
        id
            Report identifier or plain report ID.
        """

        if self.get_report_type(report_identifier=id) != 'associator':
            raise ValueError('The given report ID does not belong to an ASSOCIATOR result. ' +
                             'Please input a different ID.')

        report_id = id.report_id if isinstance(id, ReportIdentifier) else id

        result: dict[str, Any] = self.api_client.get_associator_results(group_id=self.group,
                                                                        report_id=report_id)

        actuals_version = result.pop('actuals')
        result['input'] = self.api_client.get_ts_data(self.group, actuals_version)
        return AssociatorResult(**result)

    def get_ts_versions(self, skip: int = 0, limit: int = 100) -> pd.DataFrame:
        """Gets the available time series version, ordered from newest to oldest.
            keep_until_utc shows the last day where the data is stored.

        Parameters
        ----------
        skip
            The number of initial elements of the version list to skip
        limit
            The limit on the length of the versjion list

        Returns
        -------
        Overview of the available time series versions.
        """
        results = self.api_client.get_group_ts_versions(self.group, skip, limit)
        transformed_results = []
        for version in results:
            transformed_results.append(TimeSeriesVersion(
                version_id=version['_id'],
                description=version.get('description', None),
                creation_time_utc=version.get('creation_time_utc', None),
                keep_until_utc=version['customer_specific'].get('keep_until_utc', None)
            ))
        transformed_results.sort(key=lambda x: x.creation_time_utc, reverse=True)

        return pd.DataFrame([res.model_dump() for res in transformed_results])

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
            Path to the JSON file with the CHECK-IN configuration. `config_ts_creation` and `config_checkin`
            cannot be set simultaneously. The configuration may be obtained from the last step of
            CHECK-IN using the future frontend (now.future-forecasting.de).
        file_specification
            Needed if a CSV is used with e.g. German format.

        Returns
        -------
        The identifier of the forecasting report.
        """

        assert config_fc.rerun_report_id is None, 'start_forecast_from_raw_data can not be used with rerun_report_id.'

        upload_feedback = self.upload_data(source=raw_data_source, file_specification=file_specification)

        user_input_id = upload_feedback['uuid']
        file_id = upload_feedback['files'][0]['uuid']

        res2 = self.create_time_series(user_input_id=user_input_id,
                                       file_uuid=file_id,
                                       data_definition=data_definition,
                                       config_ts_creation=config_ts_creation,
                                       config_checkin=config_checkin,
                                       file_specification=file_specification)

        version = res2['result']['tsVersion']
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

        version_data = self.api_client.get_ts_version(self.group, config.actuals_version)
        config.max_ts_len = calculate_max_ts_len(max_ts_len=config.max_ts_len,
                                                 granularity=version_data['customer_specific']['granularity'])

        if not self.is_analyst and config.db_name is not None:
            raise ValueError('Only users with the role analyst are allowed to use the parameter db_name.')

        payload = self._create_matcher_payload(config)

        result = self.api_client.execute_action(group_id=self.group,
                                                core_id=self.matcher_core_id,
                                                payload=payload,
                                                interval_status_check_in_seconds=5)
        report = ReportIdentifier.model_validate(result)
        logger.info(f'Report created with ID {report.report_id}. Matching indicators...')
        return report

    def _create_matcher_payload(self, config: MatcherConfig) -> Any:
        """Converts the MatcherConfig into the payload needed for the cov-selection core."""
        all_covs_versions = config.covs_versions
        if config.pool_covs is not None:
            pool_covs_checkin_result = self.check_in_pool_covs(requested_pool_covs=config.pool_covs)
            all_covs_versions.append(pool_covs_checkin_result.version_id)

        base_report_requested_run_status = ['Successful']
        if 'NoEvaluation' not in config.rerun_status:
            base_report_requested_run_status.append('NoEvaluation')

        config_dict: dict[str, Any] = {
            'report_description': config.title,
            'db_name': config.db_name,
            'data_config': {
                'actuals_version': config.actuals_version,
                'actuals_filter': config.actuals_filter,
                'covs_versions': all_covs_versions,
                'covs_filter': config.covs_filter,
            },
            "compute_config": {
                "evaluation_start_date": config.evaluation_start_date,
                "evaluation_end_date": config.evaluation_end_date,
                'max_ts_len': config.max_ts_len,
                "base_report_id": config.rerun_report_id,
                "base_report_requested_run_status": base_report_requested_run_status,
                "report_update_strategy": 'KEEP_OWN_RUNS',
                "cov_names": {
                    'cov_name_prefix': '',
                    'cov_name_field': 'name',
                    'cov_name_suffix': '',
                },
                "preselection": {
                    "num_obs_short_term_class": 36,
                    "max_publication_lag": config.max_publication_lag,
                },
                "postselection": {
                    "num_obs_short_term_correlation": 60,
                    "associator_report_id": config.associator_report_id,
                    "use_clustering_results": config.use_clustering_results,
                    "post_selection_queries": config.post_selection_queries,
                    "post_selection_concatenation_operator": "&",
                    "protected_selections_queries": [],
                    "protected_selections_concatenation_operator": "&"
                },
                "enable_leading_covariate_selection": config.enable_leading_covariate_selection,
                "fixed_season_length": config.fixed_season_length,
                "lag_selection": {
                    "fixed_lags": config.lag_selection.fixed_lags,
                    "min_lag": config.lag_selection.min_lag,
                    "max_lag": config.lag_selection.max_lag,
                }
            }
        }

        return {'payload': config_dict}
