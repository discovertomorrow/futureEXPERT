from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

import httpx
from authlib.jose import jwt
from pydantic import BaseModel

TOKEN_SCOPE = 'openid profile private_usage groups group_roles'


@dataclass
class FutureAuthConfig:
    """Configuration of auth."""
    auth_realm: str
    auth_server_url: str = 'https://future-auth.prognostica.de'
    auth_client_id: str = 'expert'
    token_endpoint_auth_method: str = 'none'

    @property
    def well_known_url(self) -> str:
        return f'{self.auth_server_url}/realms/{self.auth_realm}/.well-known/openid-configuration'


DEVELOPMENT_CONFIG = FutureAuthConfig(auth_realm='development')
STAGING_CONFIG = FutureAuthConfig(auth_realm='development')
PRODUCTION_CONFIG = FutureAuthConfig(auth_realm='future')

ENVIRONMENT_CONFIGS = {'production': PRODUCTION_CONFIG,
                       'staging': STAGING_CONFIG,
                       'development': DEVELOPMENT_CONFIG}

logger = logging.getLogger(__name__)


class OpenIDConfiguration(BaseModel):
    """Relevant fields from the OpenID Connect well-known configuration endpoint."""

    model_config = {'extra': 'ignore'}

    token_endpoint: str
    userinfo_endpoint: str
    jwks_uri: str
    end_session_endpoint: str


class TokenCredentials(BaseModel):
    """OAuth2 token response credentials."""

    refresh_token: str
    access_token: str
    expires_in: int
    refresh_expires_in: int


class FutureAuthClient:
    """Client for authentication in future."""

    def __init__(self,
                 environment: Optional[Literal['production', 'staging', 'development']] = None):
        """Initializer.

        Parameters
        ----------
        environment
            Which environment to use for the calculation; defaults to production.
        """
        auth_configuration = ENVIRONMENT_CONFIGS.get(environment or 'production')
        if auth_configuration is None:
            raise ValueError(
                f'Invalid environment {environment} only {list(ENVIRONMENT_CONFIGS.keys())} are valid')
        assert auth_configuration is not None
        self.auth_configuration = auth_configuration
        self.openid_configuration = OpenIDConfiguration.model_validate(
            httpx.get(self.auth_configuration.well_known_url).raise_for_status().json()
        )
        self._jwks: Any = None

    def _token_request(self, **data: Any) -> dict[str, Any]:
        data['client_id'] = self.auth_configuration.auth_client_id
        response = httpx.post(self.openid_configuration.token_endpoint, data=data)
        response.raise_for_status()
        token: dict[str, Any] = response.json()
        token['token_type'] = 'Bearer'
        token['expires_at'] = time.time() + token.get('expires_in', 300)
        logger.debug(f"Token request successful. Scopes: {token.get('scope', 'none')}")
        return token

    def token(self, username: str, password: str) -> dict[str, Any]:
        return self._token_request(grant_type='password', username=username, password=password, scope=TOKEN_SCOPE)

    def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        return self._token_request(grant_type='refresh_token', refresh_token=refresh_token, scope=TOKEN_SCOPE)

    def get_access_token(self, token: dict[str, Any]) -> str:
        return str(token['access_token'])

    def decode_token(self, access_token: str) -> dict[str, Any]:
        """Decode and verify the JWT access token."""
        if self._jwks is None:
            response = httpx.get(self.openid_configuration.jwks_uri)
            response.raise_for_status()
            self._jwks = response.json()
        return dict(jwt.decode(access_token, self._jwks))

    def get_userinfo(self, access_token: str) -> dict[str, Any]:
        """Gets the keycloak userinfo.

        Note: This requires an active Keycloak session. For offline tokens
        where the session may have expired, use decode_token() instead.
        """
        response = httpx.get(
            self.openid_configuration.userinfo_endpoint,
            headers={'Authorization': f'Bearer {access_token}'}
        )
        response.raise_for_status()
        return response.json()  # type: ignore

    def logout(self, refresh_token: str) -> None:
        """Invalidates the session by revoking the refresh token.

        Parameters
        ----------
        refresh_token
            The refresh token to revoke.
        """
        response = httpx.post(
            self.openid_configuration.end_session_endpoint,
            data={'client_id': self.auth_configuration.auth_client_id, 'refresh_token': refresh_token}
        )
        response.raise_for_status()

    def get_user_roles(self, access_token: str) -> list[str]:
        """Gets user roles."""
        decoded_token = self.decode_token(access_token)
        return decoded_token['resource_access'][self.auth_configuration.auth_client_id]['roles']  # type: ignore

    def get_user_groups(self, access_token: str) -> list[str]:
        """Gets user groups."""
        decoded_token = self.decode_token(access_token)
        return decoded_token['groups']  # type: ignore
