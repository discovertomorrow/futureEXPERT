import pytest

from futureexpert import *
from futureexpert.batch_forecast import calculate_max_ts_len, create_forecast_payload


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
    expected_error_message = f'Given max_ts_len {max_ts_len} is not allowed for granularity {granularity}. Check the variable MAX_TS_LEN_CONFIG for allowed configuration.'

    # Act
    with pytest.raises(ValueError) as excinfo:
        calculate_max_ts_len(max_ts_len, granularity)
        assert expected_error_message in str(excinfo.value)
