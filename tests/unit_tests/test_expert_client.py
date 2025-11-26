import os

import dotenv

from futureexpert.expert_client import ExpertClient


def test_init___given_refresh_token___valid_login():
    # Arrange
    client_with_user_credentials = ExpertClient.from_dotenv()
    initial_refresh_token = client_with_user_credentials.api_client.token['refresh_token']

    # Act
    client_with_token = ExpertClient(refresh_token=initial_refresh_token)

    # Assert
    assert client_with_token.api_client.token['refresh_token'] != initial_refresh_token  # successfully refreshed


def test_logout__given_user_and_pw__without_error():
    # Arrange
    dotenv.load_dotenv()
    del os.environ['FUTURE_REFRESH_TOKEN']
    client = ExpertClient()

    # Act
    client.logout()

    with ExpertClient() as client:
        print('Login successful.')
