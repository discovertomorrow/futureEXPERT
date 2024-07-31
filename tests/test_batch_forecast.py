
import logging

import pytest
from pydantic import ValidationError

from futureexpert import ForecastingConfig, MatcherConfig, PreprocessingConfig, ReportConfig
from futureexpert.batch_forecast import calculate_max_ts_len, create_forecast_payload, create_matcher_payload


def test_PreprocessingConfig___given_fixed_seasonalities_and_season_detection___stops_with_error() -> None:

    # Arrange
    expected_error_message = 'If fixed seasonalities should be used season detection must be off.'

    # Act
    with pytest.raises(ValueError) as excinfo:
        PreprocessingConfig(use_season_detection=True, fixed_seasonalities=[1, 2])
        assert expected_error_message in str(excinfo.value)


def test_create_forecast_payload___given_lower_upper_bound_None___returns_payload_with_lower_upper_bound_none() -> None:

    # Arrange
    dummy_version = '12345678'
    config = ReportConfig(title="Test", forecasting=ForecastingConfig(fc_horizon=5))

    # Act
    payload = create_forecast_payload(version=dummy_version, config=config)

    # Assert
    assert payload['payload']['forecasting']['lower_bound'] is None
    assert payload['payload']['forecasting']['upper_bound'] is None


def test_calculate_max_ts_len___given_no_ts_len___returns_default_value() -> None:

    # Arrange
    max_ts_len = None
    granularity = 'daily'

    # Act
    result = calculate_max_ts_len(max_ts_len, granularity)

    # Assert
    assert result == 365


def test_calculate_max_ts_len___given_ts_len___returns_given_ts_len() -> None:

    # Arrange
    max_ts_len = 120
    granularity = 'daily'

    # Act
    result = calculate_max_ts_len(max_ts_len, granularity)

    # Assert
    assert result == 120


def test_calculate_max_ts_len___given_ts_len_out_of_range___returns_error() -> None:

    # Arrange
    max_ts_len = 1400
    granularity = 'daily'
    expected_error_message = f'Given max_ts_len {max_ts_len} is not allowed for granularity {granularity}. ' + \
        'Check the variable MAX_TS_LEN_CONFIG for allowed configuration.'

    # Act
    with pytest.raises(ValueError) as excinfo:
        calculate_max_ts_len(max_ts_len, granularity)
        assert expected_error_message in str(excinfo.value)


def test_create_matcher_payload___given_default_configuration___returns_start_and_end_dates_with_none() -> None:
    # Arrange
    config = MatcherConfig(title="Test", actuals_version="89sadhkasgdsadffff", covs_version="89sadhkasgdsadffff")

    # Act
    payload = create_matcher_payload(config=config)

    # Assert
    assert payload['payload']['compute_config']['evaluation_start_date'] is None
    assert payload['payload']['compute_config']['evaluation_end_date'] is None


def test_create_matcher_payload___given_min_and_max_lag___returns_without_error() -> None:
    # Arrange
    config = MatcherConfig(title="Test", actuals_version="89sadhkasgdsadffff", covs_version="89sadhkasgdsadffff",
                           lag_selection_min_lag=2, lag_selection_max_lag=8)

    # Act
    create_matcher_payload(config=config)


@pytest.mark.parametrize("query", [
    ["BetterThanNoCov", "Lag <= 1", "Lag < 1"],
    ["Lag >= 1"],
    ["Lag > 1", "Lag == 1"],
    ["Lag != 1"],
    ["Lag == 11", "1 < Lag"],
    ["Lag==11", "Rank <= 1", "Rank < 1"],
    ["Rank >= 1"],
    ["Rank > 1", "Rank == 1"],
    ["Rank != 1", "Rank == 11", "1 < Rank"],
    ["Rank==11"]
])
def test_create_matcher_payload___given_valid_post_selection_parameters___returns_without_error(query: list[str]) -> None:

    # Arrange
    config = MatcherConfig(title="Test", actuals_version="89sadhkasgdsadffff", covs_version="89sadhkasgdsadffff",
                           post_selection_queries=query)

    # Assert
    assert config.post_selection_queries == query


@pytest.mark.parametrize("invalid_query", [
    ["BetterThanNonCont"],
    ["sadfasdfBetterThanNoCovasdf"],
    ["Rank*..*8.2f1f3"],
    ["Rank*.2f1f3", "asasdfLagasd==dsdf8", "asasdfRankasd==dsdf8"]
])
def test_create_matcher_payload___given_incorrect_post_selection_parameters___raises_error(invalid_query: list[str]) -> None:

    # Arrange
    with pytest.raises(ValidationError) as exc_info:
        MatcherConfig(title="Test", actuals_version="89sadhkasgdsadffff", covs_version="89sadhkasgdsadffff",
                      post_selection_queries=invalid_query)

    # Assert
    assert 'The following post-selection queries are invalidly formatted' in str(exc_info.value)


@pytest.mark.parametrize("mixed_query", [
    (["BetterThanNonCont", "Lag > 1", "Lag == 1"]),
    (["Rank==11", "sadfasdfBetterThanNoCovasdf"])
])
def test_create_matcher_payload___given_mixed_post_selection_parameters___raises_error(mixed_query: list[str]) -> None:

    # Arrange
    with pytest.raises(ValidationError) as exc_info:
        MatcherConfig(title="Test", actuals_version="89sadhkasgdsadffff", covs_version="89sadhkasgdsadffff",
                      post_selection_queries=mixed_query)

    # Assert
    assert 'The following post-selection queries are invalidly formatted' in str(exc_info.value)
