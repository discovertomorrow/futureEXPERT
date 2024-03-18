from __future__ import annotations

import io
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Tuple, Union

import requests
import tenacity
from keycloak import KeycloakOpenID
from requests import Response
from requests.exceptions import RequestException


@dataclass
class FutureConfig:
    auth_realm: str
    api_url: str
    auth_server_url: str = 'https://future-auth.prognostica.de'
    auth_client_id: str = 'frontend'


DEVELOPMENT_CONFIG = FutureConfig(api_url='https://future-dev.prognostica.de/api/v1/', auth_realm='development')
STAGING_CONFIG = FutureConfig(api_url='https://future-staging.prognostica.de/api/v1/', auth_realm='development')
PRODUCTION_CONFIG = FutureConfig(api_url='https://future.prognostica.de/api/v1/', auth_realm='future')

FUTURE_CONFIGS = {'production': PRODUCTION_CONFIG,
                  'staging': STAGING_CONFIG,
                  'development': DEVELOPMENT_CONFIG}

logger = logging.getLogger(__name__)


def get_json(response: requests.Response) -> Any:
    """Gets JSON from a success response object or raises an error.

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
        Base URL to be used. May include a path, e.g. https://future.prognostica.de/api/v1/
    url
        Relative URL path to be added to the base URL, e.g. /groups/customer1/userinputs

    Returns
    -------
    The joined URL, e.g. https://future.prognostica.de/api/v1/groups/customer1/userinputs
    """
    return f"{base.strip('/')}/{url.strip('/')}"


class NonTransientError(Exception):
    """Raised in case of an error that cannot be retried."""


class FutureApiClient:
    """Future API client."""
    def __thread_refresh(future_api: FutureApiClient) -> None:
        """Refreshes the access token."""
        time.sleep(int(future_api.token['expires_in']*.1))
        while (future_api.auto_refresh):
            logger.debug("auto token refresh")
            future_api.refresh_token()
            time.sleep(int(future_api.token['expires_in']*.7))

    def __init__(self, user: str, password: str, totp: Any = None, environment: Literal['production', 'staging', 'development'] = 'production', auto_refresh: bool = True):
        """Initializer.

        Parameters
        ----------
        user
            The username to be used to connect to the future api.
        password
            The users password.        
        totp
            Optional OTP token of the user.
        environment
            Which environment to use for the calculation, defaults to production.
        auto_refresh
            If enabled, automatically refreshes the access token in the background in a daemon thread.
        """
        future_config = FUTURE_CONFIGS.get(environment)

        if future_config is None:
            raise Exception(f'Invalid environment {environment} only {list(FUTURE_CONFIGS.keys())} are valid')
        assert future_config is not None
        self.future_config = future_config

        self.keycloak_openid = KeycloakOpenID(
            server_url=self.future_config.auth_server_url, client_id=self.future_config.auth_client_id, realm_name=self.future_config.auth_realm, verify=True)
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
                                  'retry': tenacity.retry_if_exception_type(RequestException),
                                  'reraise': True}

    @property
    def userinfo(self) -> Any:
        """Gets the keycloak userinfo."""
        return self.keycloak_openid.userinfo(self.token['access_token'])

    def _api_get_request(self, path: str, timeout: int | None = None) -> requests.Response:
        """Runs a GET request against the future api.

        Parameters
        ----------
        path
            Path of the endpoint.
        timeout
            Optional timeout of the request.

        Returns
        -------
        The response from the request.
        """
        request_url = urljoin(base=self.future_config.api_url, url=path)
        logger.debug(f'Sending GET request to {request_url}...')
        return requests.get(request_url,
                            headers={
                                'Authorization': f'Bearer {self.token["access_token"]}'},
                            timeout=timeout)

    def _api_post_request(self, path: str, json: Any, timeout: int | None = None) -> requests.Response:
        """Runs a POST request against the future api.

        Parameters
        ----------
        path
            Path of the endpoint.
        json
            JSON data to be send.
        timeout
            Optional timeout of the request.

        Returns
        -------
        The response from the request.
        """
        request_url = urljoin(base=self.future_config.api_url, url=path)
        logger.debug(f'Sending POST request to {request_url}...')
        return requests.post(request_url, headers={'Authorization': f'Bearer {self.token["access_token"]}'},
                             json=json,
                             timeout=timeout)

    def get_groups(self) -> Any:
        """Gets the groups of the current user."""
        return get_json(self._api_get_request('groups'))

    def get_user_inputs_for_group(self, group_id: str) -> Any:
        """Gets the user inputs of the given group.

        Parameters
        ----------
        group_id
            The ID of the relevant group.
        """
        return get_json(self._api_get_request(f'groups/{group_id}/userinputs'))

    def upload_user_inputs_for_group(self, group_id: str, filename: Optional[str] = None, df_file: Optional[Tuple[str, Any]] = None) -> Any:
        """Uploads the user inputs of the given group.

        Parameters
        ----------
        group_id
            The ID of the relevant group.

        Returns
        -------
            id of the user inputs
        """
        path = f'/groups/{group_id}/userinputs'
        payload: dict[str, Union[Any,  Tuple[str, Any]]]
        if filename is not None:
            payload = {'file': open(filename, 'rb')}

        if df_file is not None:
            payload = {'file': df_file}

        request_url = urljoin(base=self.future_config.api_url, url=path)
        retryer = tenacity.Retrying(**self.retry_config)
        initial_response = retryer(self._request_or_raise(request=lambda: requests.post(
            headers={'Authorization': f'Bearer {self.token["access_token"]}'},
            url=request_url, files=payload)))

        return initial_response.json()

    def get_report_status(self, group_id: str, report_id: int) -> Any:
        """Gets the report status of the given report id.

        Parameters
        ----------
        group_id
            The ID of the relevant group.
        report_id
            ID of the Report

        Returns
        -------
            Amount of each run status.
        """
        return get_json(self._api_get_request(f'groups/{group_id}/reports/{report_id}/status'))

    def get_forecasts_and_actuals(self, group_id: str, report_id: int) -> Any:
        """Collects forecasts and actuals from the database.

        Parameters
        ----------
        group_id
            The ID of the relevant group.        
        report_id
            ID of the Report
         Returns
        -------
            Actuals and Forecasts for each time series of the given report
        """

        return get_json(self._api_get_request(f'groups/{group_id}/reports/{report_id}/forecasts'))

    def get_backtesting_results(self, group_id: str, report_id: int) -> Any:
        """Collects backtesting results from the database.

        Parameters
        ----------
        group_id
            The ID of the relevant group.        
        report_id
            ID of the Report

         Returns
        -------
            Backtesting results for each time series of the given report
        """

        return get_json(self._api_get_request(f'groups/{group_id}/reports/{report_id}/backtesting'))

    def get_preprocessing_results(self, group_id: str, report_id: int) -> Any:
        """Collects preprocessing results from the database.

        Parameters
        ----------
        group_id
            The ID of the relevant group.        
        report_id
            ID of the Report

         Returns
        -------
            Preprocessing results for each time series of the given report
        """

        return get_json(self._api_get_request(f'groups/{group_id}/reports/{report_id}/preprocessing'))

    @staticmethod
    def _request_or_raise(request: Callable[[], Response]) -> Callable[[], requests.Response]:
        """Wrapper of an request to be used with tenacity retry handling.

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
        """Executes and monitors a future core action.

        Parameters
        ----------
        group_id
            The ID of the relevant group.
        core_id
            The ID of the future core to be executed.
        payload
            The payload of the request. This is not the payload of the future core.
        interval_status_check_in_seconds
            Interval in seconds between status requests while the future core action is running.
        timeout_in_seconds
            Overall timeout waiting for the future core action to have finished.
            The actual running time might be longer due to retrying requests.

        Returns
        -------
        The JSON result payload of the future core action.
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
                # 2. raised RequestException (after retries)
                # 3. Overall timeout exceeded
                status_response = retryer(self._request_or_raise(
                    lambda: self._api_get_request(action_status_path, timeout=30)))
                logger.debug(
                    f'Got status response with status code {status_response.status_code} and content:\n{str(status_response.content)}')
                status_response_payload = status_response.json()['payload']
                if status_response_payload['status'] == 'finished':
                    # stop status requests
                    break
                if status_response_payload['status'] not in ['created', 'pending', 'running']:
                    raise NonTransientError(
                        f"Job failed with status response '{status_response_payload}'.")
                time.sleep(interval_status_check_in_seconds)

            # Even if the status was not successful (timeout exceeded), try to get the result
            response = retryer(self._request_or_raise(
                request=lambda: self._api_get_request(action_path, timeout=30)))
        except RequestException as exc_info:
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
