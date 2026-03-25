from __future__ import annotations

import copy
import json
import logging
import os
import pprint
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Type, Union, cast

import httpx
import pandas as pd
from authlib.integrations.httpx_client import OAuth2Client
from cachetools import LRUCache
from dotenv import load_dotenv
from pydantic import ConfigDict, validate_call

from futureexpert._auth import FutureAuthClient
from futureexpert.associator import AssociatorConfig, AssociatorResult
from futureexpert.checkin import CheckInResult, DataDefinition, FileSpecification, TimeSeriesVersion, TsCreationConfig
from futureexpert.forecast import ForecastResult, ForecastResults, ReportConfig
from futureexpert.forecast_consistency import (ConsistentForecastMetadata,
                                               MakeForecastConsistentConfiguration,
                                               ReconciliationConfig)
from futureexpert.matcher import MatcherConfig, MatcherResult
from futureexpert.pool import CheckInPoolResult, PoolCovDefinition, PoolCovOverview
from futureexpert.shaper import ScenarioValuesConfig, ShaperConfig, ShaperResult
from futureexpert.shared_models import (ChainedReportIdentifier,
                                        Covariate,
                                        CovariateRef,
                                        PydanticModelList,
                                        ReportIdentifier,
                                        ReportStatus,
                                        ReportSummary)

pp = pprint.PrettyPrinter(indent=4)
logger = logging.getLogger(__name__)


class ExpertClient:
    """Client for the FutureEXPERT REST API.

    This client provides the same interface as futureexpert.ExpertClient but
    communicates with the expert-api REST API instead of directly with the backend.

    It can be used as a drop-in replacement for ExpertClient when you want to
    use the REST API instead of the Python SDK.
    """

    def __init__(
        self,
        refresh_token: Optional[str] = None,
        access_token: Optional[str] = None,
        group: Optional[str] = None,
        environment: Optional[Literal['production', 'staging', 'development']] = None,
        timeout: int = 300,
        max_retries: int = 3
    ) -> None:
        """Initialize the client from a token.

        If you want to login using username and password, consider using ExpertClient.from_user_password.

        Parameters
        ----------
        refresh_token
            Authentication refresh token for Bearer authentication.
            If not provided, uses environment variable FUTURE_REFRESH_TOKEN.

            You can retrieve a long-lived refresh token (offline token) in the user settings of the futureEXPERT Dashboard
            or using the Open ID Connect token endpoint of our identity provider.

            Example for calling the token endpoint with scope `offline_access`:
            curl -s -X POST "https://future-auth.prognostica.de/realms/future/protocol/openid-connect/token" \
                    -H "Content-Type: application/x-www-form-urlencoded" \
                    --data-urlencode "client_id=expert" \
                    --data-urlencode "grant_type=password" \
                    --data-urlencode "scope=openid offline_access" \
                    --data-urlencode "username=$FUTURE_USER" \
                    --data-urlencode "password=$FUTURE_PW" | jq -r .refresh_token
        access_token
            Authentication access token for Bearer authentication.

            If used instead of refresh_token, no automated token refresh is possible.
        group
            Optional group name for users in multiple groups.
            If not provided, uses environment variable FUTURE_GROUP.
        environment
            Optional environment (production, staging, development).
            If not provided, uses environment variable FUTURE_ENVIRONMENT.
        timeout
            Request timeout in seconds (default: 300)
        max_retries
            Maximum number of retries for failed requests (default: 3)
        """
        self.environment = cast(Literal['production', 'staging', 'development'],
                                environment or os.getenv('FUTURE_ENVIRONMENT') or 'production')
        self.api_url = os.getenv('EXPERT_API_URL', _EXPERT_API_URLS[self.environment]).rstrip('/')
        self.auth_client = FutureAuthClient(environment=self.environment)
        self.group = group or os.getenv('FUTURE_GROUP')

        refresh_token = refresh_token or os.getenv('FUTURE_REFRESH_TOKEN')
        if refresh_token:
            self._oauth_token = self.auth_client.refresh_token(refresh_token)
        else:
            if access_token:
                # Decode access_token token for token signature validation
                self.auth_client.decode_token(access_token)

                # A token without `refresh_token` is never tried to be refreshed in OAuth2Client.
                # A token without `expires_at` / `expires_in` is considered not expired by OAuth2Client.
                self._oauth_token = {'access_token': access_token}
            else:
                raise ValueError(
                    'A token must be provided via parameter `refresh_token` or `access_token` '
                    'or FUTURE_REFRESH_TOKEN environment variable.\nAlternatively, use `.from_user_password`.'
                )

        if not self.group:
            authorized_groups = self.auth_client.get_user_groups(self._oauth_token['access_token'])
            if len(authorized_groups) == 1:
                self.group = authorized_groups[0]
            else:
                raise ValueError(
                    f'You have access to multiple groups. Please select one of the following: {authorized_groups}')

        self.timeout = timeout
        self.max_retries = max_retries
        self.report_status_cache: LRUCache[str, ReportStatus] = LRUCache(maxsize=5)

        logger.info('Successfully logged in to futureEXPERT.')

    def _update_token(self, token: dict[str, Any], refresh_token: str = '', access_token: str = '') -> None:
        """Callback for authlib's OAuth2Client to store a refreshed token.

        The signature (token, refresh_token, access_token) is required by OAuth2Client's update_token interface.
        """
        self._oauth_token = token

    @property
    def oauth2_client(self) -> OAuth2Client:
        # Create httpx client with retry transport
        transport = httpx.HTTPTransport(retries=self.max_retries)

        return OAuth2Client(
            client_id=self.auth_client.auth_configuration.auth_client_id,
            token_endpoint=self.auth_client.openid_configuration.token_endpoint,
            token_endpoint_auth_method=self.auth_client.auth_configuration.token_endpoint_auth_method,
            token=self._oauth_token,
            update_token=self._update_token,
            leeway=30,
            base_url=self.api_url,
            timeout=self.timeout,
            transport=transport
        )

    def _request(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] = {},
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Make HTTP request to the API.

        Parameters
        ----------
        method
            HTTP method (GET, POST, etc.)
        path
            API endpoint path
        params
            Query parameters
        json_data
            JSON request body
        files
            Files for multipart upload
        data
            Form data for multipart upload

        Returns
        -------
        Response data (parsed JSON)

        Raises
        ------
        httpx.HTTPStatusError
            If the request fails
        """
        try:
            params_with_group = {**params, 'group': self.group}
            with self.oauth2_client as client:
                response = client.request(
                    method=method,
                    url=path,
                    params=params_with_group,
                    json=json_data,
                    files=files,
                    data=data
                )
            response.raise_for_status()

            # Return parsed JSON or None for empty responses
            if response.content:
                return response.json()
            return None

        except httpx.HTTPStatusError as http_status_error:
            error_mapping = {400: ValueError, 500: RuntimeError}
            error_type = error_mapping.get(http_status_error.response.status_code)
            if error_type is not None:
                try:
                    json_response = cast(dict[str, Any], http_status_error.response.json())
                    # property 'error' is contained if wrapped from a ValueError or RuntimeError
                    # property 'detail' is contained if HTTP 400 occered in FastAPI parameter deserialization
                    # property 'details' is contained if HTTP 500 on server side
                    error_message = json_response.get('details') or json_response.get('detail') or json_response.get('error')
                    assert error_message is not None, 'expecting property error or detail'
                except Exception:
                    # just logging the inner exception, but raising the outer HTTP exception in the end
                    logger.exception('Failed to handle server exception properly.')
                else:
                    # else block of the try...except statement - be careful with indentation
                    raise error_type(error_message)
            logger.error(f'API request {method} {path} failed with status code '
                         f'{http_status_error.response.status_code}: {http_status_error.response.text}')
            raise http_status_error

    # ==================== Data Upload and Check-in ====================

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def upload_data(
        self,
        source: Union[pd.DataFrame, str],
        file_specification: Optional[FileSpecification] = None
    ) -> Any:
        """Upload raw data for further processing.

        Parameters
        ----------
        source
            Path to a CSV file or a pandas DataFrame.
        file_specification
            File specification for CSV parsing.

        Returns
        -------
        Upload feedback with user_input_id and file_uuid
        """
        if isinstance(source, pd.DataFrame):
            # Convert DataFrame to JSON for upload
            data_json = source.to_dict(orient='records')
            form_data = {
                'data': json.dumps(data_json)
            }
            if file_specification:
                form_data['file_specification'] = json.dumps(file_specification.model_dump())

            return self._request('POST', '/api/v1/check-in/data', data=form_data)
        else:
            # Upload file
            with open(source, 'rb') as f:
                files = {'file': f}
                data = {}
                if file_specification:
                    data['file_specification'] = json.dumps(file_specification.model_dump())

                return self._request('POST', '/api/v1/check-in/data', files=files, data=data)

    def get_data(self) -> Any:
        """Get available raw data.

        Returns
        -------
        Meta information of the data already uploaded.
        """
        return self._request('GET', '/api/v1/check-in/data')

    @validate_call
    def check_data_definition(
        self,
        user_input_id: str,
        file_uuid: str,
        data_definition: DataDefinition,
        file_specification: FileSpecification = FileSpecification()
    ) -> Any:
        """Check data definition.

        Parameters
        ----------
        user_input_id
            UUID of the user input.
        file_uuid
            UUID of the file.
        data_definition
            Data definition specification.
        file_specification
            File specification for CSV parsing.

        Returns
        -------
        Validation result
        """
        logger.info('Started data definition using CHECK-IN...')
        payload = {
            'user_input_id': user_input_id,
            'file_uuid': file_uuid,
            'data_definition': data_definition.model_dump(),
            'file_specification': file_specification.model_dump()
        }
        result = self._request('POST', '/api/v1/check-in/validate', json_data=payload)
        logger.info('Finished data definition.')
        return result

    @validate_call
    def create_time_series(
        self,
        user_input_id: str,
        file_uuid: str,
        data_definition: Optional[DataDefinition] = None,
        config_ts_creation: Optional[TsCreationConfig] = None,
        config_checkin: Optional[str] = None,
        file_specification: FileSpecification = FileSpecification()
    ) -> Any:
        """Create time series from already uploaded data.

        This is the second step of the check-in process, after upload_data.

        Parameters
        ----------
        user_input_id
            UUID of the user input (from upload_data response).
        file_uuid
            UUID of the file (from upload_data response).
        data_definition
            Data definition specification.
        config_ts_creation
            Time series creation configuration.
        config_checkin
            Path to JSON config file (alternative to data_definition + config_ts_creation).
        file_specification
            File specification for CSV parsing.

        Returns
        -------
        Time series creation result with version information
        """
        logger.info('Creating time series using CHECK-IN...')
        form_data: Dict[str, Any] = {
            'user_input_id': user_input_id,
            'file_uuid': file_uuid,
        }

        if data_definition:
            form_data['data_definition'] = json.dumps(data_definition.model_dump())
        if config_ts_creation:
            form_data['config_ts_creation'] = json.dumps(config_ts_creation.model_dump())
        if file_specification:
            form_data['file_specification'] = json.dumps(file_specification.model_dump())

        files: Dict[str, Any] = {}
        if config_checkin:
            files['config_checkin'] = open(config_checkin, 'rb')

        try:
            result = self._request('POST', '/api/v1/check-in/create', files=files or None, data=form_data)
            logger.info('Finished time series creation.')
            return result
        finally:
            for f in files.values():
                f.close()

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def check_in_time_series(
        self,
        raw_data_source: Union[pd.DataFrame, Path, str],
        data_definition: Optional[DataDefinition] = None,
        config_ts_creation: Optional[TsCreationConfig] = None,
        config_checkin: Optional[str] = None,
        file_specification: FileSpecification = FileSpecification()
    ) -> str:
        """Check in time series data.

        Only available in `Standard`, `Premium` and `Enterprise` subscription packages.

        Parameters
        ----------
        raw_data_source
            DataFrame with raw data or path to CSV file.
        data_definition
            Data definition specification.
        config_ts_creation
            Time series creation configuration.
        config_checkin
            Path to JSON config file (alternative to data_definition + config_ts_creation).
        file_specification
            File specification for CSV parsing.

        Returns
        -------
        Version ID of the created time series
        """
        form_data: Dict[str, Any] = {}
        files: Dict[str, Any] = {}

        if data_definition:
            form_data['data_definition'] = json.dumps(data_definition.model_dump())
        if config_ts_creation:
            form_data['config_ts_creation'] = json.dumps(config_ts_creation.model_dump())
        if config_checkin:
            files['config_checkin'] = open(config_checkin, 'rb')
        if file_specification:
            form_data['file_specification'] = json.dumps(file_specification.model_dump())

        try:

            if isinstance(raw_data_source, pd.DataFrame):

                with tempfile.TemporaryDirectory() as tmpdir:
                    time_stamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
                    file_path = os.path.join(tmpdir, f'expert-{time_stamp}.csv')
                    date_format = data_definition.date_column.format if data_definition else None
                    raw_data_source.to_csv(path_or_buf=file_path, index=False, sep=file_specification.delimiter,
                                           decimal=file_specification.decimal, encoding='utf-8-sig',
                                           date_format=date_format)
                    files['file'] = open(file_path, 'rb')
                    result = self._request('POST', '/api/v1/check-in', files=files or None, data=form_data)

            else:
                files['file'] = open(raw_data_source, 'rb')
                result = self._request('POST', '/api/v1/check-in', files=files or None, data=form_data)
            return str(result['version_id'])
        finally:
            for f in files.values():
                f.close()

    @validate_call
    def check_in_pool_covs(
        self,
        requested_pool_covs: List[PoolCovDefinition],
        description: Optional[str] = None
    ) -> CheckInPoolResult:
        """Create a new version from pool covariates.

        Parameters
        ----------
        requested_pool_covs
            List of pool covariate definitions.
        description
            Short description of the selected covariates.

        Returns
        -------
        CheckInPoolResult with version_id and metadata
        """
        logger.info('Creating time series using checkin-pool...')
        payload = {
            'requested_pool_covs': [cov.model_dump() for cov in requested_pool_covs],
            'description': description
        }
        result = self._request('POST', '/api/v1/check-in/pool-covariate', json_data=payload)
        logger.info('Finished time series creation.')
        return CheckInPoolResult(**result)

    # ==================== Time Series ====================

    @validate_call
    def get_time_series(self, version_id: str) -> CheckInResult:
        """Get time series data by version ID.

        Parameters
        ----------
        version_id
            Time series version ID.

        Returns
        -------
        CheckInResult with time series data
        """
        result = self._request('GET', f'/api/v1/ts/{version_id}')
        return CheckInResult(**result)

    @validate_call
    def get_ts_versions(self, skip: int = 0, limit: int = 100) -> PydanticModelList[TimeSeriesVersion]:
        """Get list of time series versions.

        Parameters
        ----------
        skip
            Number of items to skip.
        limit
            Maximum number of items to return.

        Returns
        -------
        DataFrame with time series versions
        """
        params = {'skip': skip, 'limit': limit}
        results = self._request('GET', '/api/v1/ts', params=params)
        return PydanticModelList([TimeSeriesVersion.model_validate(raw_result) for raw_result in results])

    # ==================== Pool Covariates ====================

    @validate_call
    def get_pool_cov_overview(
        self,
        granularity: Optional[str] = None,
        search: Optional[str] = None
    ) -> PoolCovOverview:
        """Get overview of available pool covariates.

        Parameters
        ----------
        granularity
            Filter by granularity (Day or Month).
        search
            Full-text search query.

        Returns
        -------
        PoolCovOverview with available covariates
        """
        params = {}
        if granularity:
            params['granularity'] = granularity
        if search:
            params['search'] = search

        result = self._request('GET', '/api/v1/pool', params=params)
        return PoolCovOverview(overview_json=result['overview_json'])

    # ==================== Forecasting ====================

    @validate_call
    def start_forecast(
        self,
        version: str,
        config: ReportConfig,
        reconciliation_config: Optional[ReconciliationConfig] = None
    ) -> Union[ReportIdentifier, ChainedReportIdentifier]:
        """Start a forecasting report.

        Parameters
        ----------
        version
            Time series version ID.
        config
            Forecast configuration.
        reconciliation_config
            Configuration to make forecasts consistent over hierarchical levels.

        Returns
        -------
        ReportIdentifier with report_id and settings_id.
        If reconciliation_config is provided, returns ChainedReportIdentifier
        with prerequisites containing the forecast report identifier.
        """
        payload: Dict[str, Any] = {
            'version': version,
            'config': config.model_dump()
        }

        if reconciliation_config is not None:
            payload['reconciliation_config'] = reconciliation_config.model_dump()

        logger.info('Started creating FORECAST...')
        result = self._request('POST', '/api/v1/forecast', json_data=payload)

        identifier_model = ChainedReportIdentifier if 'prerequisites' in result else ReportIdentifier
        report_identifier = identifier_model.model_validate(result)
        logger.info(f'Report created with ID {report_identifier.report_id}. Forecasts are running...')
        return report_identifier

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def start_forecast_from_raw_data(self,
                                     raw_data_source: Union[pd.DataFrame, Path, str],
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

    @validate_call
    def get_fc_results(
        self,
        id: Union[ReportIdentifier, int],
        include_k_best_models: int = 1,
        include_backtesting: bool = False,
        include_discarded_models: bool = False
    ) -> ForecastResults:
        """Get forecast results.

        Parameters
        ----------
        id
            Report identifier or report ID.
        include_k_best_models
            Number of best models to include.
        include_backtesting
            Include backtesting results.
        include_discarded_models
            Include discarded models.

        Returns
        -------
        ForecastResults with forecast data
        """
        report_id = id.report_id if isinstance(id, ReportIdentifier) else id
        report_status = self.get_report_status(id=id)
        has_results = self._can_load_results(report_status)

        if not has_results:
            return ForecastResults(forecast_results=[])

        params = {
            'include_k_best_models': include_k_best_models,
            'include_backtesting': include_backtesting,
            'include_discarded_models': include_discarded_models
        }
        result = self._request('GET', f'/api/v1/forecast/{report_id}/results', params=params)

        # Parse results
        forecast_results = [ForecastResult.model_validate(r) for r in result['forecast_results']]
        fc_results = ForecastResults(forecast_results=forecast_results)

        if result.get('consistency') is not None:
            fc_results.consistency = ConsistentForecastMetadata.model_validate(result['consistency'])

        return fc_results

    # ==================== Matcher ====================

    @validate_call
    def start_matcher(self, config: MatcherConfig) -> ReportIdentifier:
        """Start a covariate matcher report.

        Parameters
        ----------
        config
            Matcher configuration.

        Returns
        -------
        ReportIdentifier with report_id and settings_id
        """
        payload = {'config': config.model_dump()}
        result = self._request('POST', '/api/v1/matcher', json_data=payload)
        report = ReportIdentifier.model_validate(result)
        logger.info(f'Report created with ID {report.report_id}. Matching indicators...')
        return report

    @validate_call
    def get_matcher_results(self, id: Union[ReportIdentifier, int]) -> List[MatcherResult]:
        """Get matcher results.

        Parameters
        ----------
        id
            Report identifier or report ID.

        Returns
        -------
        List of MatcherResult objects
        """
        report_id = id.report_id if isinstance(id, ReportIdentifier) else id

        report_status = self.get_report_status(id=id)
        has_results = self._can_load_results(report_status)

        if not has_results:
            return []

        result = self._request('GET', f'/api/v1/matcher/{report_id}/results')
        return [MatcherResult(**r) for r in result]

    # ==================== Associator ====================

    @validate_call
    def start_associator(self, config: AssociatorConfig) -> ReportIdentifier:
        """Start an associator report.

        Parameters
        ----------
        config
            Associator configuration.

        Returns
        -------
        ReportIdentifier with report_id and settings_id
        """
        payload = {'config': config.model_dump()}
        result = self._request('POST', '/api/v1/associator', json_data=payload)
        report = ReportIdentifier.model_validate(result)
        logger.info(f'Report created with ID {report.report_id}. Associator is running...')
        return report

    @validate_call
    def get_associator_results(self, id: Union[ReportIdentifier, int]) -> Optional[AssociatorResult]:
        """Get associator results.

        Parameters
        ----------
        id
            Report identifier or report ID.

        Returns
        -------
        Results of the ASSOCIATOR report.
        """
        report_id = id.report_id if isinstance(id, ReportIdentifier) else id

        report_status = self.get_report_status(id=id)
        has_results = self._can_load_results(report_status)

        if not has_results:
            return None

        result = self._request('GET', f'/api/v1/associator/{report_id}/results')
        return AssociatorResult(**result)

    # ==================== Reports ====================

    @validate_call
    def get_reports(self, skip: int = 0, limit: int = 100) -> PydanticModelList[ReportSummary]:
        """Get list of available reports.

        Parameters
        ----------
        skip
            Number of items to skip.
        limit
            Maximum number of items to return.

        Returns
        -------
        The available reports from newest to oldest.
        """
        params = {'skip': skip, 'limit': limit}
        result = self._request('GET', '/api/v1/report', params=params)
        return PydanticModelList([ReportSummary.model_validate(report) for report in result])

    @validate_call
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
        report_id = report_identifier.report_id

        cache_key = f'{report_id}_{include_error_reason}'
        if cache_key in self.report_status_cache:
            return self.report_status_cache[cache_key]

        # Determine endpoint based on report type
        report_type = self.get_report_type(report_identifier=report_id)

        # Use specific endpoint based on type
        params = {'include_error_reason': include_error_reason}
        if report_type in ['forecast', 'MongoForecastingResultSink', 'hierarchical-forecast']:
            raw_result = self._request('GET', f'/api/v1/forecast/{report_id}/status', params=params)
        elif report_type in ['matcher', 'CovariateSelection']:
            raw_result = self._request('GET', f'/api/v1/matcher/{report_id}/status', params=params)
        elif report_type == 'associator':
            raw_result = self._request('GET', f'/api/v1/associator/{report_id}/status', params=params)
        elif report_type == 'shaper':
            raw_result = self._request('GET', f'/api/v1/shaper/{report_id}/status', params=params)
        else:
            raise RuntimeError(f'Unsupported report type {report_type}')

        result = ReportStatus(**raw_result)
        if result.progress.requested == result.progress.finished and result.is_finished:
            self.report_status_cache[cache_key] = result

        return result

    @validate_call
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

    def _can_load_results(self, report_status: ReportStatus) -> bool:
        """Checks if results of an report can be returned and create log messages."""

        if report_status.progress.finished == 0:
            logger.warning('The report is not finished. No results to return.')
            return False

        if report_status.progress.finished != report_status.progress.requested:
            logger.warning('The report is not finished.')

        if report_status.result_type == 'matcher':
            if report_status.progress.finished < report_status.progress.requested and report_status.results.successful > 0:
                logger.warning('The report is not finished. Returning incomplete results.')
                return True
            if report_status.results.successful == 0:
                logger.warning('No results to return. Check `get_report_status` for details.')
                return False

        if report_status.result_type != 'matcher':
            if report_status.progress.finished < report_status.progress.requested \
                    and (report_status.results.successful > 0 or report_status.results.no_evaluation > 0):
                logger.warning('The report is not finished. Returning incomplete results.')
                return True
            if report_status.results.successful == 0 and report_status.results.no_evaluation == 0:
                logger.warning(
                    'Zero runs were successful. No results can be returned. Check `get_report_status` for details.')
                return False

        return True

    @validate_call
    def get_report_type(self, report_identifier: Union[int, ReportIdentifier]) -> str:
        """Get report type.

        Parameters
        ----------
        report_identifier
            Report ID or identifier.

        Returns
        -------
        Report type string
        """
        report_id = report_identifier.report_id if isinstance(
            report_identifier, ReportIdentifier
        ) else report_identifier

        result = self._request('GET', f'/api/v1/report/{report_id}')
        return str(result['type'])

    @validate_call
    def start_making_forecast_consistent(
        self,
        config: MakeForecastConsistentConfiguration
    ) -> ReportIdentifier:
        """Start hierarchical forecast reconciliation process.

        Makes forecasts consistent across hierarchical levels.

        Parameters
        ----------
        config
            Configuration for the reconciliation process.

        Returns
        -------
        ReportIdentifier with report_id and settings_id
        """
        payload: Dict[str, Any] = {
            'data_selection': config.data_selection.model_dump(),
            'report_note': config.report_note
        }

        if config.db_name:
            payload['db_name'] = config.db_name
        if config.reconciliation:
            payload['reconciliation'] = config.reconciliation.model_dump()

        logger.info('Started creating hierarchical reconciliation for consistent forecasts...')
        result = self._request('POST', '/api/v1/forecast/reconcile', json_data=payload)
        report = ReportIdentifier.model_validate(result)
        logger.info(f'Report created with ID {report.report_id}. Reconciliation is running...')
        return report

    @validate_call
    def create_scenario_values(self,
                               config: ScenarioValuesConfig) -> ShaperConfig:
        """Creates scenario values for covariates based on a time series and forecast horizon.

        Parameters
        ----------
        config
            Configuration for the creation of scenario values.

        Returns
        -------
        A list of Scenario objects containing high and low projections for each covariate.
        """

        payload = {'config': config.model_dump(mode='json')}
        result = self._request('POST', '/api/v1/shaper/prepare', json_data=payload)
        return ShaperConfig(**result)

    @validate_call
    def start_scenario_forecast(self, config: ShaperConfig) -> ReportIdentifier:
        """Start forecast for scenarios.

        Parameters
        ----------
        config
            Configuration for a SHAPER run.
        """
        ref_config = copy.deepcopy(config)
        for scenario in ref_config.scenarios:
            if isinstance(scenario.ts, Covariate):
                scenario.ts = CovariateRef(name=scenario.ts.ts.name, lag=scenario.ts.lag)

        payload = {'config': ref_config.model_dump(mode='json')}

        result = self._request('POST', '/api/v1/shaper', json_data=payload)
        report = ReportIdentifier.model_validate(result)
        logger.info(f'Report created with ID {report.report_id}. Shaping scenarios...')
        return report

    @staticmethod
    def from_user_password(dotenv_path: Optional[str] = None) -> ExpertClient:
        """Initialize ExpertClient from FUTURE_USER and FUTURE_PW in .env file or environment variables."""
        load_dotenv(dotenv_path=dotenv_path)
        environment = cast(Literal['production', 'staging', 'development'], os.getenv('FUTURE_ENVIRONMENT'))
        try:
            future_user = os.environ['FUTURE_USER']
        except KeyError:
            raise MissingCredentialsError('username') from None
        try:
            future_password = os.environ['FUTURE_PW']
        except KeyError:
            raise MissingCredentialsError('password') from None
        auth_client = FutureAuthClient(environment=environment)
        token = auth_client.token(future_user, future_password)
        return ExpertClient(refresh_token=token['refresh_token'], environment=environment)

    @validate_call
    def get_shaper_results(self, id: Union[ReportIdentifier, int]) -> Optional[ShaperResult]:
        """Gets the results from the given report.

        Parameters
        ----------
        id
            Report identifier or plain report ID.

        Returns
        -------
        Results of the SHAPER report.
        """
        report_id = id.report_id if isinstance(id, ReportIdentifier) else id

        report_status = self.get_report_status(id=id)
        has_results = self._can_load_results(report_status)

        if not has_results:
            return None

        result = self._request('GET', f'/api/v1/shaper/{report_id}/results')
        return ShaperResult(**result)

    def logout(self) -> None:
        """Logout from futureEXPERT.

        If logged in with a refresh token. The refresh token is revoked.
        """
        if (refresh_token := self._oauth_token.get('refresh_token')) is None:
            raise RuntimeError('Cannot logout without refresh_token')
        self.auth_client.logout(refresh_token)
        logger.info('Successfully logged out.')


class MissingCredentialsError(RuntimeError):
    def __init__(self, missing_credential_type: str) -> None:
        super().__init__(f'Please enter {missing_credential_type} either when ' +
                         'initializing the expert client or in the the .env file!')


_EXPERT_API_URLS = {'production': 'https://expert.future-forecasting.de',
                    'staging': 'https://expert.staging.future-forecasting.de',
                    'development': 'https://expert.dev.future-forecasting.de'}
