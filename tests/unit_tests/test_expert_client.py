import time
from typing import Optional

import pytest
from authlib.jose.errors import BadSignatureError, DecodeError

from futureexpert.expert_client import ExpertClient
from futureexpert.shared_models import (ErrorReason,
                                        ReportIdentifier,
                                        ReportStatus,
                                        ReportStatusProgress,
                                        ReportStatusResults)


def test_from_user_password___valid_credentials___login_and_logout_without_errors(caplog):
    # Act
    client = ExpertClient.from_user_password()
    client.logout()

    # Assert
    assert 'Successfully logged in' in caplog.text
    assert 'Successfully logged out' in caplog.text


def test_can_load_results___no_finished_results___returns_false(caplog, expert_client: ExpertClient):
    # Arrange
    status = ReportStatus(
        id=ReportIdentifier(report_id=123, settings_id=456),
        description='Test Report',
        result_type='forecast',
        progress=ReportStatusProgress(requested=10, pending=10, finished=0),
        results=ReportStatusResults(successful=0, no_evaluation=0, error=0),
    )

    # Act
    result = expert_client._can_load_results(status)

    # Assert
    assert result is False
    assert 'The report is not finished. No results to return.' in caplog.text


def test_can_load_results___partial_results_with_some_successful___returns_true(caplog, expert_client: ExpertClient):
    # Arrange
    status = ReportStatus(
        id=ReportIdentifier(report_id=123, settings_id=456),
        description='Test Report',
        result_type='forecast',
        progress=ReportStatusProgress(requested=10, pending=5, finished=5),
        results=ReportStatusResults(successful=3, no_evaluation=0, error=2),
    )

    # Act
    result = expert_client._can_load_results(status)

    # Assert
    assert result is True
    assert 'The report is not finished. Returning incomplete results.' in caplog.text


def test_can_load_results___all_finished_all_successful___returns_true(caplog, expert_client: ExpertClient):
    # Arrange
    status = ReportStatus(
        id=ReportIdentifier(report_id=123, settings_id=456),
        description='Test Report',
        result_type='forecast',
        progress=ReportStatusProgress(requested=10, pending=0, finished=10),
        results=ReportStatusResults(successful=8, no_evaluation=2, error=0),
    )

    # Act
    result = expert_client._can_load_results(status)

    # Assert
    assert result is True
    assert len(caplog.records) == 0


def test_can_load_results___matcher_no_successful___returns_false(caplog, expert_client: ExpertClient):
    # Arrange
    status = ReportStatus(
        id=ReportIdentifier(report_id=123, settings_id=456),
        description='Test Report',
        result_type='matcher',
        progress=ReportStatusProgress(requested=10, pending=0, finished=10),
        results=ReportStatusResults(successful=0, no_evaluation=5, error=5),
    )

    # Act
    result = expert_client._can_load_results(status)

    # Assert
    assert result is False
    assert 'No results to return. Check `get_report_status` for details.' in caplog.text


def test_can_load_results___non_matcher_all_errors___returns_false(caplog, expert_client: ExpertClient):
    # Arrange
    status = ReportStatus(
        id=ReportIdentifier(report_id=123, settings_id=456),
        description='Test Report',
        result_type='forecast',
        progress=ReportStatusProgress(requested=10, pending=0, finished=10),
        results=ReportStatusResults(successful=0, no_evaluation=0, error=10),
    )

    # Act
    result = expert_client._can_load_results(status)

    # Assert
    assert result is False
    assert 'Zero runs were successful. No results can be returned.' in caplog.text


def test_can_load_results___non_matcher_some_successful_some_errors___returns_true(caplog, expert_client: ExpertClient):
    # Arrange
    status = ReportStatus(
        id=ReportIdentifier(report_id=123, settings_id=456),
        description='Test Report',
        result_type='forecast',
        progress=ReportStatusProgress(requested=10, pending=0, finished=10),
        results=ReportStatusResults(successful=5, no_evaluation=0, error=5),
    )

    # Act
    result = expert_client._can_load_results(status)

    # Assert
    assert result is True
    assert len(caplog.records) == 0


def test_can_load_results___matcher_some_successful___returns_true(caplog, expert_client: ExpertClient):
    # Arrange
    status = ReportStatus(
        id=ReportIdentifier(report_id=123, settings_id=456),
        description='Test Report',
        result_type='matcher',
        progress=ReportStatusProgress(requested=10, pending=0, finished=10),
        results=ReportStatusResults(successful=5, no_evaluation=2, error=3),
    )

    # Act
    result = expert_client._can_load_results(status)

    # Assert
    assert result is True
    assert len(caplog.records) == 0


def test_can_load_results___associator_all_errors___returns_false(caplog, expert_client: ExpertClient):
    # Arrange
    status = ReportStatus(
        id=ReportIdentifier(report_id=123, settings_id=456),
        description='Test Report',
        result_type='associator',
        progress=ReportStatusProgress(requested=5, pending=0, finished=5),
        results=ReportStatusResults(successful=0, no_evaluation=0, error=5),
    )

    # Act
    result = expert_client._can_load_results(status)

    # Assert
    assert result is False
    assert 'Zero runs were successful. No results can be returned.' in caplog.text


def test_can_load_results___finished_not_equal_requested___logs_warning(caplog, expert_client: ExpertClient):
    # Arrange
    status = ReportStatus(
        id=ReportIdentifier(report_id=123, settings_id=456),
        description='Test Report',
        result_type='forecast',
        progress=ReportStatusProgress(requested=10, pending=5, finished=5),
        results=ReportStatusResults(successful=0, no_evaluation=0, error=5),
    )

    # Act
    result = expert_client._can_load_results(status)

    # Assert
    assert result is False
    assert 'The report is not finished.' in caplog.text


def test_init___valid_access_token___no_error(expert_client: ExpertClient, monkeypatch):
    # Arrange
    valid_access_token = expert_client._oauth_token['access_token']
    monkeypatch.delenv('FUTURE_REFRESH_TOKEN')

    # Act
    client = ExpertClient(access_token=valid_access_token)

    # Assert
    client.get_data()


def test_get_data___refresh_token_available_and_expired_access_token___refreshes_token():
    # Arrange
    # do not use the client from conftest because we are going to patch it
    expert_client = ExpertClient()
    now = time.time()
    old_token = expert_client._oauth_token
    old_token['expires_at'] = now  # token expires now

    # Act
    expert_client.get_data()

    # Assert
    new_token = expert_client._oauth_token
    assert new_token['expires_at'] > now
    assert new_token['access_token'] != old_token['access_token']
    assert new_token['refresh_token'] != old_token['refresh_token']


@pytest.mark.parametrize("group", [None, 'gitlab-ci-futureexpert'])
def test_init___expired_access_token___no_error(group: Optional[str], monkeypatch):
    # Arrange
    monkeypatch.delenv('FUTURE_REFRESH_TOKEN', raising=False)
    monkeypatch.delenv('FUTURE_GROUP', raising=False)
    invalid_access_token = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJVRlRLLUZXVEhocmMwS0Y1ZXVaXzB5blMwWEpJQXVFUHFrb2c5WW5hSEV3In0.eyJleHAiOjE3NzE4NDYzNDcsImlhdCI6MTc3MTg0NDU0NywianRpIjoib25ydHJvOjg1OWJhYzRiLWQ4OTUtMzBjMy1mMjc2LWFiNjdjYzgwYTQxNCIsImlzcyI6Imh0dHBzOi8vZnV0dXJlLWF1dGgucHJvZ25vc3RpY2EuZGUvcmVhbG1zL2RldmVsb3BtZW50IiwiYXVkIjpbInJlYWxtLW1hbmFnZW1lbnQiLCJyZWN5Y2FyaW8iLCJiYWNrZW5kIiwiZnJvbnRlbmQiLCJhY2NvdW50Il0sInN1YiI6IjE5YTI0ZTFhLTBhYzYtNDU3Ny1hYzg0LWRkNmY5OGE0NGRjNyIsInR5cCI6IkJlYXJlciIsImF6cCI6ImV4cGVydCIsInNpZCI6IjBjMGZmYjVjLWQwZmMtNGZhZi04NGE1LWUwNTkyMjllZTk0NiIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsImRlZmF1bHQtcm9sZXMtZGV2ZWxvcG1lbnQiLCJmdXR1cmUtY3VzdG9tZXIiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImZ1dHVyZS1kYXRhc2NpZW50aXN0Il19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiZXhwZXJ0Ijp7InJvbGVzIjpbImFuYWx5c3QiLCJ1c2VyIl19LCJyZWFsbS1tYW5hZ2VtZW50Ijp7InJvbGVzIjpbInZpZXctdXNlcnMiLCJxdWVyeS1ncm91cHMiLCJxdWVyeS11c2VycyJdfSwicmVjeWNhcmlvIjp7InJvbGVzIjpbInVzZXIiXX0sImJhY2tlbmQiOnsicm9sZXMiOlsiYW5hbHlzdCIsInVzZXIiXX0sImZyb250ZW5kIjp7InJvbGVzIjpbImtiIiwiYW5hbHlzdCIsInVzZXIiXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgcHJpdmF0ZV91c2FnZSBncm91cHMgZ3JvdXBfcm9sZXMiLCJwcml2YXRlX3VzYWdlIjpmYWxzZSwiZ3JvdXBfcm9sZXMiOlsiZXhwbG9yYXRpb246ZXhwZXJ0IiwiZTJlLXRlc3QtZXhwZXJ0OmV4cGVydCIsImJld2VyYmVyOmZjLWFwcCIsInNuZWFrLXBlZWstMjAyNTAzMTE6ZXhwZXJ0IiwiY3VzdG9tZXIxOmV4cGVydCIsImN1c3RvbWVyMjpleHBlcnQiLCJnaXRsYWItY2ktZnV0dXJlZXhwZXJ0OmV4cGVydCIsImdyb3VwLWV4cGVydC1wcmVtaXVtOmV4cGVydCIsImJyLWRldjpib3NjaC1hbmFseXN0IiwiZ3JvdXAtZXhwZXJ0OmV4cGVydCIsInNuZWFrLXBlYWstMjAyNDExMDQ6ZXhwZXJ0IiwiZTJlLXRlc3QtMjpmYy1hcHAiLCJkZW1hbmQtcGxhbm5pbmctZGVtbzpleHBlcnQiLCJzbmVhay1wZWVrLTIwMjUwNjI3OmV4cGVydCIsImdyb3VwLWZjLWFwcDpmYy1hcHAiLCJjY2Y1ZjQ1Ny1iMzI4LTRkYWEtOTA5YS1iZTJjMzFiNGJiNGM6ZXhwZXJ0IiwiZTJlLXRlc3Qtc3Vic2NyaXB0aW9uOmV4cGVydCIsImdyb3VwLWV4cGVydC1iYXNpczpleHBlcnQiLCJncm91cC1leHBlcnQtc3RhbmRhcmQ6ZXhwZXJ0Il0sIm5hbWUiOiJDaHJpc3RpYW4gR3JvdGhlZXIiLCJncm91cHMiOlsiYmV3ZXJiZXIiLCJici1kZXYiLCJjY2Y1ZjQ1Ny1iMzI4LTRkYWEtOTA5YS1iZTJjMzFiNGJiNGMiLCJjdXN0b21lcjEiLCJjdXN0b21lcjIiLCJkZW1hbmQtcGxhbm5pbmctZGVtbyIsImUyZS10ZXN0LTEiLCJlMmUtdGVzdC0yIiwiZTJlLXRlc3QtZXhwZXJ0IiwiZTJlLXRlc3QtZnJlZSIsImUyZS10ZXN0LXN1YnNjcmlwdGlvbiIsImV4cGxvcmF0aW9uIiwiZ2l0bGFiLWNpLWZ1dHVyZWV4cGVydCIsImdyb3VwLWV4cGVydCIsImdyb3VwLWV4cGVydC1iYXNpcyIsImdyb3VwLWV4cGVydC1wcmVtaXVtIiwiZ3JvdXAtZXhwZXJ0LXN0YW5kYXJkIiwiZ3JvdXAtZmMtYXBwIiwiZ3JvdXAtZmlsZXRyYXkiLCJsYWdvLXRlc3RzIiwic25lYWstcGVhay0yMDI0MTEwNCIsInNuZWFrLXBlZWstMjAyNTAzMTEiLCJzbmVhay1wZWVrLTIwMjUwNjI3Il0sInByZWZlcnJlZF91c2VybmFtZSI6ImNocmlzdGlhbiIsImdpdmVuX25hbWUiOiJDaHJpc3RpYW4iLCJsb2NhbGUiOiJlbiIsImZhbWlseV9uYW1lIjoiR3JvdGhlZXIifQ.NcDu8IYzPwWkRoEmbuUsQNUN2HrlUUjcGQ93sgJ8MHujHffZE5Kv09IXrGzz3DMUSqCe3oRj1tWO1LiFEfuGetjyMyzKXqfv8VaE0zQ4Mz9o-tfEZXVQ-3LowQl368P_uFDkHjje8UFm7QD-HZOhDUJshumhHAhG9dJXXTgzK9dhh_jgwAxYGRWqt1hTJaWjshm4L7TTW80Nq27yEvA6xVQ6DfhdVhq4PkH7iKCk-plqXBPHznHLK5UwsxfI499h1K3iN7smwa0NnFO62JFggI3IFtIaxrLvaML0ch5rlYQ4evDZ-BZHuvDf1eJVCJjR6iyH8872xw8W_W7ik6V9vw'

    # Act
    try:
        ExpertClient(access_token=invalid_access_token, group=group)
    except ValueError as exc_info:
        if 'access to multiple groups' in str(exc_info):
            # successfully decoded expired token
            pass
        else:
            raise


def test_get_data___expired_access_token___error(monkeypatch):
    # Arrange
    monkeypatch.delenv('FUTURE_REFRESH_TOKEN', raising=False)
    invalid_access_token = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJVRlRLLUZXVEhocmMwS0Y1ZXVaXzB5blMwWEpJQXVFUHFrb2c5WW5hSEV3In0.eyJleHAiOjE3NzE4NDYzNDcsImlhdCI6MTc3MTg0NDU0NywianRpIjoib25ydHJvOjg1OWJhYzRiLWQ4OTUtMzBjMy1mMjc2LWFiNjdjYzgwYTQxNCIsImlzcyI6Imh0dHBzOi8vZnV0dXJlLWF1dGgucHJvZ25vc3RpY2EuZGUvcmVhbG1zL2RldmVsb3BtZW50IiwiYXVkIjpbInJlYWxtLW1hbmFnZW1lbnQiLCJyZWN5Y2FyaW8iLCJiYWNrZW5kIiwiZnJvbnRlbmQiLCJhY2NvdW50Il0sInN1YiI6IjE5YTI0ZTFhLTBhYzYtNDU3Ny1hYzg0LWRkNmY5OGE0NGRjNyIsInR5cCI6IkJlYXJlciIsImF6cCI6ImV4cGVydCIsInNpZCI6IjBjMGZmYjVjLWQwZmMtNGZhZi04NGE1LWUwNTkyMjllZTk0NiIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsImRlZmF1bHQtcm9sZXMtZGV2ZWxvcG1lbnQiLCJmdXR1cmUtY3VzdG9tZXIiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImZ1dHVyZS1kYXRhc2NpZW50aXN0Il19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiZXhwZXJ0Ijp7InJvbGVzIjpbImFuYWx5c3QiLCJ1c2VyIl19LCJyZWFsbS1tYW5hZ2VtZW50Ijp7InJvbGVzIjpbInZpZXctdXNlcnMiLCJxdWVyeS1ncm91cHMiLCJxdWVyeS11c2VycyJdfSwicmVjeWNhcmlvIjp7InJvbGVzIjpbInVzZXIiXX0sImJhY2tlbmQiOnsicm9sZXMiOlsiYW5hbHlzdCIsInVzZXIiXX0sImZyb250ZW5kIjp7InJvbGVzIjpbImtiIiwiYW5hbHlzdCIsInVzZXIiXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgcHJpdmF0ZV91c2FnZSBncm91cHMgZ3JvdXBfcm9sZXMiLCJwcml2YXRlX3VzYWdlIjpmYWxzZSwiZ3JvdXBfcm9sZXMiOlsiZXhwbG9yYXRpb246ZXhwZXJ0IiwiZTJlLXRlc3QtZXhwZXJ0OmV4cGVydCIsImJld2VyYmVyOmZjLWFwcCIsInNuZWFrLXBlZWstMjAyNTAzMTE6ZXhwZXJ0IiwiY3VzdG9tZXIxOmV4cGVydCIsImN1c3RvbWVyMjpleHBlcnQiLCJnaXRsYWItY2ktZnV0dXJlZXhwZXJ0OmV4cGVydCIsImdyb3VwLWV4cGVydC1wcmVtaXVtOmV4cGVydCIsImJyLWRldjpib3NjaC1hbmFseXN0IiwiZ3JvdXAtZXhwZXJ0OmV4cGVydCIsInNuZWFrLXBlYWstMjAyNDExMDQ6ZXhwZXJ0IiwiZTJlLXRlc3QtMjpmYy1hcHAiLCJkZW1hbmQtcGxhbm5pbmctZGVtbzpleHBlcnQiLCJzbmVhay1wZWVrLTIwMjUwNjI3OmV4cGVydCIsImdyb3VwLWZjLWFwcDpmYy1hcHAiLCJjY2Y1ZjQ1Ny1iMzI4LTRkYWEtOTA5YS1iZTJjMzFiNGJiNGM6ZXhwZXJ0IiwiZTJlLXRlc3Qtc3Vic2NyaXB0aW9uOmV4cGVydCIsImdyb3VwLWV4cGVydC1iYXNpczpleHBlcnQiLCJncm91cC1leHBlcnQtc3RhbmRhcmQ6ZXhwZXJ0Il0sIm5hbWUiOiJDaHJpc3RpYW4gR3JvdGhlZXIiLCJncm91cHMiOlsiYmV3ZXJiZXIiLCJici1kZXYiLCJjY2Y1ZjQ1Ny1iMzI4LTRkYWEtOTA5YS1iZTJjMzFiNGJiNGMiLCJjdXN0b21lcjEiLCJjdXN0b21lcjIiLCJkZW1hbmQtcGxhbm5pbmctZGVtbyIsImUyZS10ZXN0LTEiLCJlMmUtdGVzdC0yIiwiZTJlLXRlc3QtZXhwZXJ0IiwiZTJlLXRlc3QtZnJlZSIsImUyZS10ZXN0LXN1YnNjcmlwdGlvbiIsImV4cGxvcmF0aW9uIiwiZ2l0bGFiLWNpLWZ1dHVyZWV4cGVydCIsImdyb3VwLWV4cGVydCIsImdyb3VwLWV4cGVydC1iYXNpcyIsImdyb3VwLWV4cGVydC1wcmVtaXVtIiwiZ3JvdXAtZXhwZXJ0LXN0YW5kYXJkIiwiZ3JvdXAtZmMtYXBwIiwiZ3JvdXAtZmlsZXRyYXkiLCJsYWdvLXRlc3RzIiwic25lYWstcGVhay0yMDI0MTEwNCIsInNuZWFrLXBlZWstMjAyNTAzMTEiLCJzbmVhay1wZWVrLTIwMjUwNjI3Il0sInByZWZlcnJlZF91c2VybmFtZSI6ImNocmlzdGlhbiIsImdpdmVuX25hbWUiOiJDaHJpc3RpYW4iLCJsb2NhbGUiOiJlbiIsImZhbWlseV9uYW1lIjoiR3JvdGhlZXIifQ.NcDu8IYzPwWkRoEmbuUsQNUN2HrlUUjcGQ93sgJ8MHujHffZE5Kv09IXrGzz3DMUSqCe3oRj1tWO1LiFEfuGetjyMyzKXqfv8VaE0zQ4Mz9o-tfEZXVQ-3LowQl368P_uFDkHjje8UFm7QD-HZOhDUJshumhHAhG9dJXXTgzK9dhh_jgwAxYGRWqt1hTJaWjshm4L7TTW80Nq27yEvA6xVQ6DfhdVhq4PkH7iKCk-plqXBPHznHLK5UwsxfI499h1K3iN7smwa0NnFO62JFggI3IFtIaxrLvaML0ch5rlYQ4evDZ-BZHuvDf1eJVCJjR6iyH8872xw8W_W7ik6V9vw'

    # need to set group becaude the access token contains multiple groups
    client = ExpertClient(access_token=invalid_access_token, group='gitlab-ci-futureexpert')

    # Act & Assert
    with pytest.raises(expected_exception=RuntimeError, match='401 Unauthorized'):
        client.get_data()


@pytest.mark.parametrize("group", [None, 'gitlab-ci-futureexpert'])
def test_init___invalid_access_token___raises_decode_error(group: Optional[str], monkeypatch):
    # Arrange
    monkeypatch.delenv('FUTURE_REFRESH_TOKEN', raising=False)
    monkeypatch.delenv('FUTURE_GROUP', raising=False)
    invalid_access_token = 'foo'

    # Act & Assert
    with pytest.raises(expected_exception=DecodeError):
        ExpertClient(access_token=invalid_access_token, group=group)


@pytest.mark.parametrize("group", [None, 'gitlab-ci-futureexpert'])
def test_init___modified_access_token___raises_bad_signature_error(group: Optional[str], monkeypatch):
    # Arrange
    monkeypatch.delenv('FUTURE_REFRESH_TOKEN', raising=False)
    monkeypatch.delenv('FUTURE_GROUP', raising=False)
    invalid_access_token = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJVRlRLLUZXVEhocmMwS0Y1ZXVaXzB5blMwWEpJQXVFUHFrb2c5WW5hSEV3In0.eyJleHAiOjE3NzE4NDYzNDcsImlhdCI6MTc3MTg0NDU0NywianRpIjoib25ydHJvOjg1OWJhYzRiLWQ4OTUtMzBjMy1mMjc2LWFiNjdjYzgwYTQxNCIsImlzcyI6Imh0dHBzOi8vZnV0dXJlLWF1dGgucHJvZ25vc3RpY2EuZGUvcmVhbG1zL2RldmVsb3BtZW50IiwiYXVkIjpbInJlYWxtLW1hbmFnZW1lbnQiLCJyZWN5Y2FyaW8iLCJiYWNrZW5kIiwiZnJvbnRlbmQiLCJhY2NvdW50Il0sInN1YiI6IjE5YTI0ZTFhLTBhYzYtNDU3Ny1hYzg0LWRkNmY5OGE0NGRjNyIsInR5cCI6IkJlYXJlciIsImF6cCI6ImV4cGVydCIsInNpZCI6IjBjMGZmYjVjLWQwZmMtNGZhZi04NGE1LWUwNTkyMjllZTk0NiIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsImRlZmF1bHQtcm9sZXMtZGV2ZWxvcG1lbnQiLCJmdXR1cmUtY3VzdG9tZXIiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImZ1dHVyZS1kYXRhc2NpZW50aXN0Il19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiZXhwZXJ0Ijp7InJvbGVzIjpbImFuYWx5c3QiLCJ1c2VyIl19LCJyZWFsbS1tYW5hZ2VtZW50Ijp7InJvbGVzIjpbInZpZXctdXNlcnMiLCJxdWVyeS1ncm91cHMiLCJxdWVyeS11c2VycyJdfSwicmVjeWNhcmlvIjp7InJvbGVzIjpbInVzZXIiXX0sImJhY2tlbmQiOnsicm9sZXMiOlsiYW5hbHlzdCIsInVzZXIiXX0sImZyb250ZW5kIjp7InJvbGVzIjpbImtiIiwiYW5hbHlzdCIsInVzZXIiXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgcHJpdmF0ZV91c2FnZSBncm91cHMgZ3JvdXBfcm9sZXMiLCJwcml2YXRlX3VzYWdlIjpmYWxzZSwiZ3JvdXBfcm9sZXMiOlsiZXhwbG9yYXRpb246ZXhwZXJ0IiwiZTJlLXRlc3QtZXhwZXJ0OmV4cGVydCIsImJld2VyYmVyOmZjLWFwcCIsInNuZWFrLXBlZWstMjAyNTAzMTE6ZXhwZXJ0IiwiY3VzdG9tZXIxOmV4cGVydCIsImN1c3RvbWVyMjpleHBlcnQiLCJnaXRsYWItY2ktZnV0dXJlZXhwZXJ0OmV4cGVydCIsImdyb3VwLWV4cGVydC1wcmVtaXVtOmV4cGVydCIsImJyLWRldjpib3NjaC1hbmFseXN0IiwiZ3JvdXAtZXhwZXJ0OmV4cGVydCIsInNuZWFrLXBlYWstMjAyNDExMDQ6ZXhwZXJ0IiwiZTJlLXRlc3QtMjpmYy1hcHAiLCJkZW1hbmQtcGxhbm5pbmctZGVtbzpleHBlcnQiLCJzbmVhay1wZWVrLTIwMjUwNjI3OmV4cGVydCIsImdyb3VwLWZjLWFwcDpmYy1hcHAiLCJjY2Y1ZjQ1Ny1iMzI4LTRkYWEtOTA5YS1iZTJjMzFiNGJiNGM6ZXhwZXJ0IiwiZTJlLXRlc3Qtc3Vic2NyaXB0aW9uOmV4cGVydCIsImdyb3VwLWV4cGVydC1iYXNpczpleHBlcnQiLCJncm91cC1leHBlcnQtc3RhbmRhcmQ6ZXhwZXJ0Il0sIm5hbWUiOiJDaHJpc3RpYW4gR3JvdGhlZXIiLCJncm91cHMiOlsiYmV3ZXJiZXIiLCJici1kZXYiLCJjY2Y1ZjQ1Ny1iMzI4LTRkYWEtOTA5YS1iZTJjMzFiNGJiNGMiLCJjdXN0b21lcjEiLCJjdXN0b21lcjIiLCJkZW1hbmQtcGxhbm5pbmctZGVtbyIsImUyZS10ZXN0LTEiLCJlMmUtdGVzdC0yIiwiZTJlLXRlc3QtZXhwZXJ0IiwiZTJlLXRlc3QtZnJlZSIsImUyZS10ZXN0LXN1YnNjcmlwdGlvbiIsImV4cGxvcmF0aW9uIiwiZ2l0bGFiLWNpLWZ1dHVyZWV4cGVydCIsImdyb3VwLWV4cGVydCIsImdyb3VwLWV4cGVydC1iYXNpcyIsImdyb3VwLWV4cGVydC1wcmVtaXVtIiwiZ3JvdXAtZXhwZXJ0LXN0YW5kYXJkIiwiZ3JvdXAtZmMtYXBwIiwiZ3JvdXAtZmlsZXRyYXkiLCJsYWdvLXRlc3RzIiwic25lYWstcGVhay0yMDI0MTEwNCIsInNuZWFrLXBlZWstMjAyNTAzMTEiLCJzbmVhay1wZWVrLTIwMjUwNjI3Il0sInByZWZlcnJlZF91c2VybmFtZSI6ImNocmlzdGlhbiIsImdpdmVuX25hbWUiOiJDaHJpc3RpYW4iLCJsb2NhbGUiOiJlbiIsImZhbWlseV9uYW1lIjoiR3JvdGhlZXIifQ.NcDu8IYzPwWkRoEmbuUsQNUN2HrlUUjcGQ93sgJ8MHujHffZE5Kv09IXrGzz3DMUSqCe3oRj1tWO1LiFEfuGetjyMyzKXqfv8VaE0zQ4Mz9o-tfEZXVQ-3LowQl368P_uFDkHjje8UFm7QD-HZOhDUJshumhHAhG9dJXXTgzK9dhh_jgwAxYGRWqt1hTJaWjshm4L7TTW80Nq27yEvA6xVQ6DfhdVhq4PkH7iKCk-plqXBPHznHLK5UwsxfI499h1K3iN7smwa0NnFO62JFggI3IFtIaxrLvaML0ch5rlYQ4evDZ-BZHuvDf1eJVCJjR6iyH8872xw8W_W7ik6V9vW'

    # Act & Assert
    with pytest.raises(expected_exception=BadSignatureError):
        ExpertClient(access_token=invalid_access_token, group=group)
