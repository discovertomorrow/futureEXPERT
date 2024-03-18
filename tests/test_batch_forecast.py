import pytest

from futureexpert import *
from futureexpert.batch_forecast import create_forecast_payload


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
