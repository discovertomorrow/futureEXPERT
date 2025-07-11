from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Tuple, Union, cast

import httpx
import tenacity
from keycloak import KeycloakOpenID


@dataclass
class FutureConfig:
    auth_realm: str
    api_url: str
    auth_server_url: str = 'https://future-auth.prognostica.de'
    auth_client_id: str = 'frontend'


DEVELOPMENT_CONFIG = FutureConfig(api_url='https://api.dev.future-forecasting.de/api/v1/', auth_realm='development')
STAGING_CONFIG = FutureConfig(api_url='https://api.staging.future-forecasting.de/api/v1/', auth_realm='development')
PRODUCTION_CONFIG = FutureConfig(api_url='https://api.future-forecasting.de/api/v1/', auth_realm='future')

FUTURE_CONFIGS = {'production': PRODUCTION_CONFIG,
                  'staging': STAGING_CONFIG,
                  'development': DEVELOPMENT_CONFIG}

logger = logging.getLogger(__name__)


def get_json(response: httpx.Response) -> Any:
    """Gets JSON from a successful response object or raises an error.

    Parameters
    ----------
    response
        Response to be processed.
    """
    response.raise_for_status()
    return response.json()


def urljoin(base: str, url: str) -> str:
    """Joins an URL from a base URL and a relative path.

    In contrast to urllib.urljoin this supports a path in the base URL.

    Parameters
    ----------
    base
        Base URL to be used. May include a path, e.g. https://api.future-forecasting.de/api/v1/
    url
        Relative URL path to be added to the base URL, e.g. /groups/customer1/userinputs

    Returns
    -------
    The joined URL, e.g. https://api.future-forecasting.de/api/v1/groups/customer1/userinputs
    """
    return f"{base.strip('/')}/{url.strip('/')}"


class NonTransientError(Exception):
    """Raised in the case of an error that cannot be retried."""


class FutureApiClient:
    """_future_ API client."""
    def __thread_refresh(future_api: FutureApiClient) -> None:
        """Refreshes the access token."""
        time.sleep(int(future_api.token['expires_in']*.1))
        while (future_api.auto_refresh):
            logger.debug("auto token refresh")
            future_api.refresh_token()
            time.sleep(int(future_api.token['expires_in']*.7))

    def __init__(self,
                 user: str,
                 password: str,
                 totp: Any = None,
                 environment: Literal['production', 'staging', 'development'] = 'production',
                 auto_refresh: bool = True):
        """Initializer.

        Parameters
        ----------
        user
            The username to be used to connect to the _future_ API.
        password
            The user's password.
        totp
            Optional OTP token of the user.
        environment
            Which environment to use for the calculation; defaults to production.
        auto_refresh
            If enabled, automatically refreshes the access token in the background in a daemon thread.
        """
        future_config = FUTURE_CONFIGS.get(environment)

        if future_config is None:
            raise ValueError(f'Invalid environment {environment} only {list(FUTURE_CONFIGS.keys())} are valid')
        assert future_config is not None
        self.future_config = future_config

        self.keycloak_openid = KeycloakOpenID(
            server_url=self.future_config.auth_server_url,
            client_id=self.future_config.auth_client_id,
            realm_name=self.future_config.auth_realm, verify=True)
        self.token = self.keycloak_openid.token(user, password, totp=totp)
        self.auto_refresh = auto_refresh
        if auto_refresh:
            self._thread = threading.Thread(
                target=FutureApiClient.__thread_refresh, args=(self,))
            self._thread.daemon = True
            self._thread.start()
        self.retry_config: Any = {'stop': tenacity.stop.stop_after_attempt(max_attempt_number=7),
                                  'wait': tenacity.wait.wait_exponential(multiplier=1, exp_base=2,),
                                  'before': tenacity.before.before_log(logger, logging.DEBUG),
                                  'after': tenacity.after.after_log(logger, logging.DEBUG),
                                  'retry': tenacity.retry_if_exception_type(httpx.RequestError),
                                  'reraise': True}

    @property
    def userinfo(self) -> Any:
        """Gets the keycloak userinfo."""
        return self.keycloak_openid.userinfo(self.token['access_token'])

    @property
    def user_roles(self) -> Any:
        """Gets user role for the frontend."""
        KEYCLOAK_PUBLIC_KEY = "-----BEGIN PUBLIC KEY-----\n" + self.keycloak_openid.public_key() + "\n-----END PUBLIC KEY-----"
        options = {"verify_signature": False, "verify_aud": False, "verify_exp": False}
        decoded_token = self.keycloak_openid.decode_token(
            self.token['access_token'], key=KEYCLOAK_PUBLIC_KEY, options=options)
        return decoded_token['resource_access']['frontend']['roles']

    def _api_get_request(self,
                         path: str,
                         params: Optional[dict[str, Any]] = None,
                         timeout: Optional[int] = None) -> httpx.Response:
        """Submits a GET request to the _future_ API.

        Parameters
        ----------
        path
            Path of the endpoint.
        params
            Parameter to be send.
        timeout
            Optional timeout of the request.

        Returns
        -------
        The response from the request.
        """
        request_url = urljoin(base=self.future_config.api_url, url=path)
        logger.debug(f'Sending GET request to {request_url}...')
        return httpx.get(request_url,
                         params=params,
                         headers={
                             'Authorization': f'Bearer {self.token["access_token"]}'},
                         timeout=timeout)

    def _api_post_request(self, path: str, json: Any, timeout: Optional[int] = None) -> httpx.Response:
        """Submits a POST request to the _future_ API.

        Parameters
        ----------
        path
            Path of the endpoint.
        json
            JSON data to be sent.
        timeout
            Optional timeout of the request.

        Returns
        -------
        The response from the request.
        """
        request_url = urljoin(base=self.future_config.api_url, url=path)
        logger.debug(f'Sending POST request to {request_url}...')
        return httpx.post(request_url, headers={'Authorization': f'Bearer {self.token["access_token"]}'},
                          json=json,
                          timeout=timeout)

    def get_groups(self) -> Any:
        """Gets the groups of the current user."""
        return get_json(self._api_get_request('groups'))

    def get_group_reports(self, group_id: str, skip: int = 0, limit: int = 100) -> Any:
        """Gets the reports for the given group. The reports are ordered from newest to oldest.

        Parameters
        ----------
        group_id
            The ID of the group.
        skip
            The number reports to skip before returning.
        limit
            The maximum number of reports to return. By default 100.

        Returns
        -------
            A list of reports in the given group from newest to oldest.
        """
        params = {'skip': skip, 'limit': limit}
        return get_json(self._api_get_request(f'groups/{group_id}/reports', params=params))

    def get_user_inputs_for_group(self, group_id: str) -> Any:
        """Gets the user inputs of the given group.

        Parameters
        ----------
        group_id
            The ID of the relevant group.
        """
        return get_json(self._api_get_request(f'groups/{group_id}/userinputs'))

    def upload_user_inputs_for_group(self,
                                     group_id: str,
                                     filename: Optional[str] = None,
                                     df_file: Optional[Tuple[str, Any]] = None) -> Any:
        """Uploads the user inputs of the given group.

        Parameters
        ----------
        group_id
            The ID of the relevant group.

        Returns
        -------
            ID of the user inputs.
        """
        path = f'/groups/{group_id}/userinputs'
        payload: dict[str, Union[Any,  Tuple[str, Any]]]
        if filename is not None:
            payload = {'file': open(filename, 'rb')}

        if df_file is not None:
            payload = {'file': df_file}

        request_url = urljoin(base=self.future_config.api_url, url=path)
        retryer = tenacity.Retrying(**self.retry_config)
        initial_response = retryer(self._request_or_raise(request=lambda: httpx.post(
            headers={'Authorization': f'Bearer {self.token["access_token"]}'},
            url=request_url, files=payload)))

        return initial_response.json()

    def get_ts_version(self, group_id: str, version_id: str) -> Any:
        """Get version data.

        Parameters
        ----------
        group_id
            The ID of the relevant group.
        version_id
            The version of time series.

        Returns
        -------
        """
        return get_json(self._api_get_request(f'groups/{group_id}/ts/versions/{version_id}'))

    def get_ts_data(self, group_id: str, version_id: str) -> Any:
        """Get time series data.

        Parameters
        ----------
        group_id
            The ID of the relevant group.
        version_id
            The version of time series.
        """
        final_result = []
        batch_count = 0
        limit = 100

        while True:
            skip = batch_count * limit
            params = {'query': '{"version": {"$oid":"' + version_id + '"}}', 'skip': skip, 'limit': limit}
            response = self._api_get_request(f'groups/{group_id}/ts/values', params=params)

            if batch_count > 0 and response.status_code == 404:
                # Already got all results in the previous batch.
                break
            results = get_json(response)
            final_result.extend(results)
            if len(results) < limit:
                # Got all results with the current batch.
                break
            batch_count += 1

        return final_result

    def get_group_ts_versions(self, group_id: str, skip: int = 0, limit: int = 100) -> Any:
        """Get time series versions of the group.

        Parameters
        ----------
        group_id
            The ID of the relevant group.
        skip
            The number of versions to skip before returning.
        limit
            The maximum number of versions to return. By default 100.

        Returns
        -------
        The available time series versions.
        """
        params = {'skip': skip, 'limit': limit}
        return get_json(self._api_get_request(f'groups/{group_id}/ts/versions', params=params))

    def get_report_type(self, group_id: str, report_id: int) -> str:
        """Gets the report type of the given report ID.

        Parameters
        ----------
        group_id
            The ID of the relevant group.
        report_id
            ID of the Report

        Returns
        -------
            String representation of the type of one report.
        """
        raw_result = get_json(self._api_get_request(f'groups/{group_id}/reports/{report_id}'))
        return cast(str, raw_result['result_type'])

    def get_report_status(self, group_id: str, report_id: int, include_error_reason: bool) -> Any:
        """Gets the report status of the given report ID.

        Parameters
        ----------
        group_id
            The ID of the relevant group.
        report_id
            ID of the Report
        include_error_reason
            Determines whether log messages are to be included in the result.

        Returns
        -------
            Amount of each run status.
        """
        params = {'include_error_reason': include_error_reason}
        return get_json(self._api_get_request(f'groups/{group_id}/reports/{report_id}/status', params=params))

    def get_fc_results(self, group_id: str, report_id: int, include_k_best_models: int, include_backtesting: bool) -> Any:
        """Retrieves forecasts and actuals from the database.

        Parameters
        ----------
        group_id
            The ID of the relevant group.
        report_id
            ID of the Report
        include_k_best_models
            Number of k best models for which results are to be returned.
        include_backtesting
            Should backtesting results are to be returned.
        Returns
        -------
        Actuals and forecasts for each time series in the given report.
        """
        params = {'include_k_best_models': include_k_best_models, 'include_backtesting': include_backtesting}
        return get_json(self._api_get_request(f'groups/{group_id}/reports/{report_id}/results/fc', params=params))

    def get_pool_cov_overview(self,
                              granularity: Optional[str] = None,
                              search: Optional[str] = None) -> Any:
        """Gets an overview of all covariates available on POOL according to the given filters.

        Parameters
        ----------
        granularity
            If set, returns only data matching that granularity (Day or Month).
        search
            If set, performs a full-text search and only returns data found
            in that search.

        Returns
        ------
        Covariate overview data.
        """
        params = {}
        if granularity:
            params['distance'] = granularity
        if search:
            params['search'] = search
        path = 'indicators'
        response = self._api_get_request(params=params, path=path, timeout=30)
        if len(response_json := get_json(response)) == 0:
            raise ValueError('No data found with the specified parameters')
        return response_json

    def get_matcher_results(self, group_id: str, report_id: int) -> Any:
        """Collects covariate matcher results from the database.

        Parameters
        ----------
        group_id
            The ID of the relevant group.
        report_id
            ID of the report.
        Returns
        -------
        Actuals and covariate ranking.
        """

        return get_json(self._api_get_request(f'groups/{group_id}/reports/{report_id}/results/cov-selection'))

    @staticmethod
    def _request_or_raise(request: Callable[[], httpx.Response]) -> Callable[[], httpx.Response]:
        """Wraps a request to be used with tenacity retry handling.

        request
            Callable of the request to be performed.

        Returns
        -------
        A callable that handles the response status.
        """
        def func() -> Any:
            response = request()
            if response.status_code == 404:
                raise NonTransientError("Not found (404)")
            response.raise_for_status()
            return response
        return func

    def execute_action(self,
                       group_id: str,
                       core_id: str,
                       payload: dict[str, Any],
                       interval_status_check_in_seconds: int,
                       timeout_in_seconds: int = 3600) -> Any:
        """Executes and monitors a futureCORE action.

        Parameters
        ----------
        group_id
            The ID of the relevant group.
        core_id
            The ID of the futureCORE to be executed.
        payload
            The payload of the request. This is not the payload of the futureCORE.
        interval_status_check_in_seconds
            Interval in seconds between status requests while the futureCORE action is running.
        timeout_in_seconds
            Overall timeout waiting for the futureCORE action to have finished.
            The actual running time might be longer due to retrying httpx.

        Returns
        -------
        The JSON result payload of the futureCORE action.
        """
        path = f'/groups/{group_id}/cores/{core_id}/actions'
        logger.debug(f'Request payload:\n{payload}')
        try:
            retryer = tenacity.Retrying(**self.retry_config)

            initial_response = retryer(self._request_or_raise(
                request=lambda: self._api_post_request(path, json=payload, timeout=30)))

            action_id = initial_response.json()['actionId']
            action_path = f'groups/{group_id}/cores/{core_id}/actions/{action_id}'
            action_status_path = urljoin(action_path, 'status')

            for _ in range(math.ceil(timeout_in_seconds/interval_status_check_in_seconds)):
                # There are three ways out of this loop:
                # 1. result json property 'status' is not 'created', 'pending', 'running' or 'unknown'
                # 2. raised RequestError (after retries)
                # 3. Overall timeout exceeded
                status_response = retryer(self._request_or_raise(
                    lambda: self._api_get_request(action_status_path, timeout=30)))
                logger.debug(
                    f'Got response with status code {status_response.status_code} and content:\n{str(status_response.content)}')
                status_response_payload = status_response.json()['payload']
                if status_response_payload['status'] == 'finished':
                    # stop status requests
                    break
                if status_response_payload['status'] not in ['created', 'pending', 'running']:

                    response = retryer(self._request_or_raise(
                        request=lambda: self._api_get_request(action_path, timeout=30)))

                    raise NonTransientError(
                        f"Job failed with status response'{status_response_payload}' and job response:\n{str(response.json()['payload'])}.")
                time.sleep(interval_status_check_in_seconds)

            # Even if the status was not successful (timeout exceeded), try to get the result
            response = retryer(self._request_or_raise(
                request=lambda: self._api_get_request(action_path, timeout=30)))
        except httpx.RequestError as exc_info:
            logger.error(msg=str(exc_info), exc_info=True)
            raise NonTransientError(
                f'Request failed caused by {str(exc_info)}') from exc_info
        logger.debug(
            f'Got response with status code {response.status_code} and content:\n{str(response.content)}')
        return response.json()['payload']

    def refresh_token(self) -> None:
        """Gets the refresh token."""
        self.token = self.keycloak_openid.refresh_token(
            self.token['refresh_token'])
