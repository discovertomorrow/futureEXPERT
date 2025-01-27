import pandas as pd
import pytest

from futureexpert import MAX_TS_LEN_CONFIG, ExpertClient, ForecastingConfig, PreprocessingConfig, ReportConfig
from futureexpert._helpers import calculate_max_ts_len, remove_arima_if_not_allowed
from futureexpert.forecast import (export_forecasts_to_pandas,
                                   export_forecasts_with_overview_to_pandas,
                                   export_result_overview_to_pandas)


def test_PreprocessingConfig___given_fixed_seasonalities_and_season_detection___stops_with_error() -> None:

    # Arrange
    expected_error_message = 'If fixed seasonalities is enabled, then season detection must be off'

    # Act
    with pytest.raises(ValueError, match=expected_error_message):
        PreprocessingConfig(use_season_detection=True, fixed_seasonalities=[1, 2])


def test_create_forecast_payload___given_lower_upper_bound_None___returns_payload_with_lower_upper_bound_none() -> None:

    # Arrange
    dummy_version = '12345678'
    config = ReportConfig(title="Test", forecasting=ForecastingConfig(fc_horizon=5))
    client = ExpertClient.from_dotenv()

    # Act
    payload = client._create_forecast_payload(version=dummy_version, config=config)

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
    assert result == MAX_TS_LEN_CONFIG['daily']


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
        'Check the environment variable MAX_TS_LEN_CONFIG for allowed configuration.'

    # Act
    with pytest.raises(ValueError, match=expected_error_message):
        calculate_max_ts_len(max_ts_len, granularity)


def test_remove_arima_if_not_allowed__given_daily_granularity___returns_no_arima() -> None:

    # Arrange
    forecasting_methods = ['ARIMA', 'Naive']
    granularity = 'daily'

    # Act
    result = remove_arima_if_not_allowed(granularity, forecasting_methods)

    # Assert
    assert len(result) == 1
    assert result[0] == 'Naive'


def test_remove_arima_if_not_allowed__given_monthly_granularity___returns_complete_list() -> None:

    # Arrange
    forecasting_methods = ['ARIMA', 'Naive']
    granularity = 'monthly'

    # Act
    result = remove_arima_if_not_allowed(granularity, forecasting_methods)

    # Assert
    assert len(result) == 2
    assert result == forecasting_methods


def test_remove_arima_if_not_allowed__given_empty_forecasting_methods___returns_complete_list() -> None:

    # Arrange
    forecasting_methods = []
    granularity = 'monthly'

    # Act
    result = remove_arima_if_not_allowed(granularity, forecasting_methods)

    # Assert
    assert len(result) == 0


def test_remove_arima_if_not_allowed__given_only_arima_as_forecasting_methods___raises_error() -> None:

    # Arrange
    forecasting_methods = ['ARIMA']
    granularity = 'weekly'

    # Act
    with pytest.raises(ValueError, match='ARIMA is not supported for granularities below monthly.'):
        remove_arima_if_not_allowed(granularity, forecasting_methods)


def test_export_result_overview_to_pandas___given_simple_results___runs_without_error(
        sample_fc_result_1,
        sample_fc_result_2,
        sample_fc_result_3):
    sample_results = [sample_fc_result_1,
                      sample_fc_result_2,
                      sample_fc_result_3]

    df = export_result_overview_to_pandas(sample_results)
    assert isinstance(df, pd.DataFrame)


def test_export_result_overview_to_pandas___given_simple_results___is_of_expected_structure(
        sample_fc_result_1,
        sample_fc_result_2,
        sample_fc_result_3):
    sample_results = [sample_fc_result_1,
                      sample_fc_result_2,
                      sample_fc_result_3]

    df = export_result_overview_to_pandas(sample_results)
    assert df.shape == (3, 12)
    assert set(df.columns.to_list()) == {'name', 'level', 'model', 'cov', 'cov_lag', 'season_length', 'ts_class',
                                         'quantization', 'trend', 'recent_trend', 'missing_values_count', 'outliers_count'}


def test_export_forecasts_to_pandas___given_simple_results___is_of_expected_structure(
        sample_fc_result_1,
        sample_fc_result_2,
        sample_fc_result_3):
    sample_results = [sample_fc_result_1,
                      sample_fc_result_2,
                      sample_fc_result_3]

    df = export_forecasts_to_pandas(sample_results)
    assert df.shape == (9, 5)
    assert set(df.columns.to_list()) == {'name', 'time_stamp_utc',
                                         'point_forecast_value', 'lower_limit_value', 'upper_limit_value'}


def test_export_forecasts_with_metadata___given_simple_results___is_of_expected_structure(
        sample_fc_result_1,
        sample_fc_result_2,
        sample_fc_result_3):
    sample_results = [sample_fc_result_1,
                      sample_fc_result_2,
                      sample_fc_result_3]

    df = export_forecasts_with_overview_to_pandas(sample_results)
    assert df.shape == (9, 16)
    assert set(df.columns.to_list()) == {'name', 'level', 'model', 'cov', 'cov_lag', 'season_length', 'ts_class',
                                         'quantization', 'trend', 'recent_trend', 'missing_values_count', 'outliers_count',
                                         'time_stamp_utc', 'point_forecast_value', 'lower_limit_value', 'upper_limit_value'}
