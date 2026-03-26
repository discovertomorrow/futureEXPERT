import logging

import pandas as pd
import pydantic
import pytest
from conftest import create_forecast_result

from futureexpert import MAX_TS_LEN_CONFIG, ForecastingConfig, MethodSelectionConfig, PreprocessingConfig, ReportConfig
from futureexpert._helpers import calculate_max_ts_len
from futureexpert.forecast import ForecastResult, ForecastResults


def test_PreprocessingConfig___given_fixed_seasonalities_and_season_detection___stops_with_error() -> None:
    # Arrange
    expected_error_message = 'If fixed seasonalities is enabled, then season detection must be off'

    # Act & Assert
    with pytest.raises(ValueError, match=expected_error_message):
        PreprocessingConfig(use_season_detection=True, fixed_seasonalities=[1, 2])


def test_PreprocessingConfig___given_auto_few_obs_and_missing_changepoint_detection___stops_with_error() -> None:
    # Arrange
    expected_error_message = 'If phase_out_method is set to AUTO_FEW_OBS, then detect_changepoints must be on.'

    # Act & Assert
    with pytest.raises(ValueError, match=expected_error_message):
        PreprocessingConfig(detect_outliers=False, phase_out_method='AUTO_FEW_OBS')


def test_PreprocessingConfig___given_missing_recent_trend_num_obs_and_missing_num_seasons___stops_with_error() -> None:
    # Arrange
    expected_error_message = 'Both recent_trend_num_observations and recent_trend_num_seasons cannot be None at the same time.'

    # Act & Assert
    with pytest.raises(ValueError, match=expected_error_message):
        PreprocessingConfig(recent_trend_num_observations=None, recent_trend_num_seasons=None)


def test_ReportConfig___given_changed_phase_out_fc_methods_and_missing_phase_out_detection___logs_warning(caplog) -> None:
    # Arrange
    expected_error_message = ('Phase-out detection must be enabled in PreprocessingConfig'
                              ' so changes in phase_out_fc_methods in MethodSelectionConfig take effect.')
    preprocessing = PreprocessingConfig(detect_outliers=False)
    method_selection = MethodSelectionConfig(phase_out_fc_methods=['AutoArima'])

    # Act
    with caplog.at_level(logging.WARNING):
        ReportConfig(title='Test', forecasting=ForecastingConfig(fc_horizon=5),
                     preprocessing=preprocessing, method_selection=method_selection)

    # Assert
    assert expected_error_message in caplog.messages


def test_ReportConfig___given_empty_phase_out_fc_methods_and_active_phase_out_detection___stops_with_error() -> None:
    # Arrange
    expected_error_message = 'Phase out forecasting method cannot be empty when phase out detection is enabled.'
    preprocessing = PreprocessingConfig(phase_out_method='TRAILING_ZEROS')
    method_selection = MethodSelectionConfig(phase_out_fc_methods=[])

    # Act & Assert
    with pytest.raises(ValueError, match=expected_error_message):
        ReportConfig(title='Test', forecasting=ForecastingConfig(fc_horizon=5),
                     preprocessing=preprocessing, method_selection=method_selection)


def test_MethodSelectionConfig___with_equal_coverage_bt_strategy_and_shift_len_greater_one___stops_in_error() -> None:
    # Arrange
    shift_length = 2
    expected_error_message = 'Equal-Coverage-Backtesting-Strategy only allows a shift length of 1.'

    # Act & Assert
    with pytest.raises(ValueError, match=expected_error_message):
        MethodSelectionConfig(
            backtesting_strategy='equal_coverage',
            shift_len=shift_length
        )


def test_MethodSelectionConfig___with_empty_dict_in_step_weights___results_in_error() -> None:
    # Arrange
    expected_error_message = 'Empty dictionary for step_weights is not allowed.'

    # Act & Assert
    with pytest.raises(ValueError, match=expected_error_message):
        MethodSelectionConfig(
            step_weights={}
        )


def test_method_selection_config___with_empty_methods_list_for_hierarchy_level___raises_error() -> None:
    # Arrange
    forecasting_methods_per_hierarchy_level = {
        0: ['Naive', 'AutoEsCov'],
        1: []  # Empty list not allowed
    }

    # Act & Assert
    with pytest.raises(pydantic.ValidationError, match='List should have at least 1 item'):
        MethodSelectionConfig(
            forecasting_methods_per_hierarchy_level=forecasting_methods_per_hierarchy_level
        )


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

    # Act & Assert
    with pytest.raises(ValueError, match=expected_error_message):
        calculate_max_ts_len(max_ts_len, granularity)


def test_add_grouping_columns_to_overview___no_name_clash___adds_keys_directly() -> None:
    # Arrange
    overview = {'name': 'ts1', 'model': 'ARIMA'}
    grouping = {'region': 'EU', 'category': 'A'}

    # Act
    ForecastResult._add_grouping_columns_to_overview(overview=overview,
                                                     reserved_keywords=overview.keys(),
                                                     grouping=grouping)

    # Assert
    assert overview == {'name': 'ts1', 'model': 'ARIMA', 'region': 'EU', 'category': 'A'}


def test_add_grouping_columns_to_overview___name_clash___adds_prefix_to_all_grouping_keys(caplog) -> None:
    # Arrange
    overview = {'name': 'ts1', 'model': 'ARIMA'}
    grouping = {'name': 'clashing_value', 'region': 'EU'}

    # Act
    with caplog.at_level(logging.WARNING):
        ForecastResult._add_grouping_columns_to_overview(overview=overview,
                                                         reserved_keywords=overview.keys(),
                                                         grouping=grouping)

    # Assert
    assert overview == {'name': 'ts1', 'model': 'ARIMA', 'grouping-name': 'clashing_value', 'grouping-region': 'EU'}
    assert any('grouping-' in message for message in caplog.messages)


def test_export_result_overview_to_pandas___given_simple_results___runs_without_error(
        sample_fc_result_1,
        sample_fc_result_2,
        sample_fc_result_3):
    # Arrange
    sample_results = ForecastResults(forecast_results=[sample_fc_result_1,
                                                       sample_fc_result_2,
                                                       sample_fc_result_3])

    # Act
    df = sample_results.export_result_overview_to_pandas()

    # Assert
    assert isinstance(df, pd.DataFrame)


def test_export_result_overview_to_pandas___given_simple_results___is_of_expected_structure(
        sample_fc_result_1,
        sample_fc_result_2,
        sample_fc_result_3):
    # Arrange
    sample_results = ForecastResults(forecast_results=[sample_fc_result_1,
                                                       sample_fc_result_2,
                                                       sample_fc_result_3])

    # Act
    df = sample_results.export_result_overview_to_pandas()

    # Assert
    assert df.shape == (3, 12)
    assert set(df.columns.to_list()) == {'name', 'level', 'model', 'cov', 'cov_lag', 'season_length', 'ts_class',
                                         'quantization', 'trend', 'recent_trend', 'missing_values_count', 'outliers_count'}


def test_export_result_overview_to_pandas___given_clashing_grouping___prefixes_grouping_columns_and_preserves_expected_ordering():
    # Arrange
    fc_result = create_forecast_result(actuals_name='actuals_clash', model_name='ARIMA',
                                       season_length=6, ts_class='smooth', quantization=2)
    fc_result.input.actuals.grouping = {'name': 'clashing_value', 'region': 'EU'}
    sample_results = ForecastResults(forecast_results=[fc_result])

    # Act
    df = sample_results.export_result_overview_to_pandas()

    # Assert
    assert 'name' in df.columns
    assert df['name'].iloc[0] == 'actuals_clash'
    assert 'grouping-name' in df.columns
    assert df['grouping-name'].iloc[0] == 'clashing_value'
    assert 'grouping-region' in df.columns
    assert len(df.columns) > 3 # more columns than 'name', 'grouping-name' and 'grouping-region'
    assert df.columns[:3].tolist() == ['name', 'grouping-name', 'grouping-region'] # name and groupings first


def test_export_forecasts_to_pandas___given_simple_results___is_of_expected_structure(
        sample_fc_result_1,
        sample_fc_result_2,
        sample_fc_result_3):
    # Arrange
    sample_results = ForecastResults(forecast_results=[sample_fc_result_1,
                                                       sample_fc_result_2,
                                                       sample_fc_result_3])

    # Act
    df = sample_results.export_forecasts_to_pandas()

    # Assert
    assert df.shape == (9, 5)
    assert set(df.columns.to_list()) == {'name', 'time_stamp_utc',
                                         'point_forecast_value', 'lower_limit_value', 'upper_limit_value'}


def test_export_forecasts_with_metadata___given_simple_results___is_of_expected_structure(
        sample_fc_result_1,
        sample_fc_result_2,
        sample_fc_result_3):
    # Arrange
    sample_results = ForecastResults(forecast_results=[sample_fc_result_1,
                                                       sample_fc_result_2,
                                                       sample_fc_result_3])

    # Act
    df = sample_results.export_forecasts_with_overview_to_pandas()

    # Assert
    assert df.shape == (9, 16)
    assert set(df.columns.to_list()) == {'name', 'level', 'model', 'cov', 'cov_lag', 'season_length', 'ts_class',
                                         'quantization', 'trend', 'recent_trend', 'missing_values_count', 'outliers_count',
                                         'time_stamp_utc', 'point_forecast_value', 'lower_limit_value', 'upper_limit_value'}
