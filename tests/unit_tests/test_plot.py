from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from futureexpert.forecast import (BacktestingValue,
                                   ChangedValue,
                                   ChangePoint,
                                   ComparisonDetails,
                                   ForecastValue,
                                   MissingValue,
                                   Model,
                                   ModelStatus,
                                   Outlier,
                                   Plausibility,
                                   RankingDetails)
from futureexpert.plot import (_add_level_shifts,
                               _calculate_few_observation_borders,
                               _calculate_replaced_missing_borders,
                               _fill_missing_values_for_plot,
                               _prepare_backtesting,
                               _prepare_characteristics)
from futureexpert.shared_models import PositiveInt, ValidatedPositiveInt

granularities = ['yearly', 'quarterly', 'monthly', 'weekly', 'daily', 'hourly', 'halfhourly']


def create_example_df(granularity: str) -> pd.DataFrame:
    freq_map = {
        'yearly': 'YS',
        'quarterly': 'QS',
        'monthly': 'MS',
        'weekly': 'W',
        'daily': 'D',
        'hourly': 'h',
        'halfhourly': '30min'
    }
    date_range = pd.date_range(start='2020-01-01', periods=20, freq=freq_map[granularity])
    rng = np.random.default_rng(12345)

    return pd.DataFrame({
        'date': date_range,
        'actuals': rng.random(20)
    })


@pytest.mark.parametrize('granularity', granularities)
def test_fill_missing_values___no_missing_values___df_stays_unchanged(granularity):
    # Arrange
    df = create_example_df(granularity)
    # Act
    result_df = _fill_missing_values_for_plot(granularity, df)

    # Assert
    assert result_df.equals(df)


@pytest.mark.parametrize('granularity', granularities)
def test_fill_missing_values___with_missing_data___fills_missing_data_with_nan(granularity):
    # Arrange
    df = create_example_df(granularity)
    missing_indices = [2, 3, 7, 9, 11]
    df.drop(missing_indices, inplace=True)

    # Act
    result_df = _fill_missing_values_for_plot(granularity, df)

    # Assert
    assert result_df.shape[0] == 20, f'Expected 20 rows for {granularity}, but got {result_df.shape[0]}.'
    assert result_df.loc[missing_indices, 'actuals'].isna().all()


def test_calculate_few_observation_borders__with_start__returns_expected_borders():
    # Arrange
    start_date = datetime(2023, 1, 1)
    actuals = [start_date + timedelta(days=i) for i in range(70)]
    few_observations = [ChangePoint(time_stamp_utc=datetime(2023, 2, 1), change_point_type='FEW_OBSERVATIONS_LEFT'),
                        ChangePoint(time_stamp_utc=datetime(2023, 2, 20), change_point_type='FEW_OBSERVATIONS_LEFT'),
                        ChangePoint(time_stamp_utc=datetime(2023, 2, 10), change_point_type='FEW_OBSERVATIONS_RIGHT')]

    # Act
    result = _calculate_few_observation_borders(actuals, few_observations)

    # Assert
    assert result[0][0] == start_date
    assert result[0][1] == datetime(2023, 1, 31)
    assert result[1][0] == datetime(2023, 2, 10)
    assert result[1][1] == datetime(2023, 2, 19)


def test_calculate_few_observation_borders__with_end__returns_expected_borders():
    # Arrange
    start_date = datetime(2023, 1, 1)
    actuals = [start_date + timedelta(days=i) for i in range(70)]
    few_observations = [ChangePoint(time_stamp_utc=datetime(2023, 3, 1), change_point_type='FEW_OBSERVATIONS_RIGHT')]

    # Act
    result = _calculate_few_observation_borders(actuals, few_observations)

    # Assert
    assert result[0][0] == datetime(2023, 3, 1)
    assert result[0][1] == datetime(2023, 3, 11)


def test_calculate_few_observation_borders__with_single_time_frame__returns_expected_borders():
    # Arrange
    start_date = datetime(2023, 1, 1)
    actuals = [start_date + timedelta(days=i) for i in range(70)]
    few_observations = [ChangePoint(time_stamp_utc=datetime(2023, 2, 1), change_point_type='FEW_OBSERVATIONS_RIGHT'),
                        ChangePoint(time_stamp_utc=datetime(2023, 3, 1), change_point_type='FEW_OBSERVATIONS_LEFT')]

    # Act
    result = _calculate_few_observation_borders(actuals, few_observations)

    # Assert
    assert result[0][0] == datetime(2023, 2, 1)
    assert result[0][1] == datetime(2023, 2, 28)


def test_calculate_few_observation_borders__with_multiple_time_frame__returns_expected_borders():
    # Arrange
    start_date = datetime(2023, 1, 1)
    actuals = [start_date + timedelta(days=i) for i in range(150)]
    few_observations = [ChangePoint(time_stamp_utc=datetime(2023, 2, 1), change_point_type='FEW_OBSERVATIONS_RIGHT'),
                        ChangePoint(time_stamp_utc=datetime(2023, 3, 1), change_point_type='FEW_OBSERVATIONS_LEFT'),
                        ChangePoint(time_stamp_utc=datetime(2023, 4, 1), change_point_type='FEW_OBSERVATIONS_RIGHT'),
                        ChangePoint(time_stamp_utc=datetime(2023, 4, 15), change_point_type='FEW_OBSERVATIONS_LEFT')]

    # Act
    result = _calculate_few_observation_borders(actuals, few_observations)

    # Assert
    assert len(result) == 2
    assert result[0][0] == datetime(2023, 2, 1)
    assert result[0][1] == datetime(2023, 2, 28)
    assert result[1][0] == datetime(2023, 4, 1)
    assert result[1][1] == datetime(2023, 4, 14)


def test_calculate_add_level_shifts__with_multiple_shifts__returns_expected_level_shifts():
    # Arrange
    start_date = datetime(2023, 1, 1)
    actuals = [start_date + timedelta(days=i) for i in range(25)]
    df_ac = pd.DataFrame({'date': actuals,
                          'actuals': [1] * 5 + [5] * 10 + [20] * 10})
    level_shifts = [ChangePoint(time_stamp_utc=datetime(2023, 1, 6), change_point_type='LEVEL_SHIFT'),
                    ChangePoint(time_stamp_utc=datetime(2023, 1, 16), change_point_type='LEVEL_SHIFT')]

    # Act
    result = _add_level_shifts(df_ac, level_shifts)

    print(result)

    # Assert
    assert result.level_shift.sum() == 255
    assert result.loc[0:4, 'level_shift'].sum() == 5
    assert result.loc[5:14, 'level_shift'].sum() == 50
    assert result.loc[15:25, 'level_shift'].sum() == 200


def test_calculate_replaced_missing_borders___given_valid_input_with_edge_cases___returns_expected_borders():
    # Arrange
    df = create_example_df('monthly')

    df['replaced_missing'] = np.nan
    df.loc[0:4, 'replaced_missing'] = 4

    df.loc[10:12, 'replaced_missing'] = 4

    df.loc[17:18, 'replaced_missing'] = 4

    # Act
    result = _calculate_replaced_missing_borders(df)

    # Assert
    assert len(result) == 3
    assert result[0][0] == datetime(2020, 1, 1)
    assert result[0][1] == datetime(2020, 6, 1)
    assert result[1][0] == datetime(2020, 10, 1)
    assert result[1][1] == datetime(2021, 2, 1)
    assert result[2][0] == datetime(2021, 5, 1)
    assert result[2][1] == datetime(2021, 8, 1)


def test_calculate_replaced_missing_borders___given_valid_input___returns_expected_borders():
    # Arrange
    df = create_example_df('monthly')

    df['replaced_missing'] = np.nan
    df.loc[10:12, 'replaced_missing'] = 4

    # Act
    result = _calculate_replaced_missing_borders(df)

    # Assert
    assert len(result) == 1
    assert result[0][0] == datetime(2020, 10, 1)
    assert result[0][1] == datetime(2021, 2, 1)


def test__prepare_characteristics___none_of_values_are_in_timeframe___all_values_are_removed_temporarily(sample_fc_result_1):
    # Arrange
    df = create_example_df('monthly')
    fc_result = sample_fc_result_1
    dates = [datetime(2000, 1, 1), datetime(2000, 2, 1)]
    outliers = [Outlier(time_stamp_utc=x, original_value=1) for x in dates]
    change_points = [ChangePoint(time_stamp_utc=x, change_point_type='LEVEL_SHIFT') for x in dates]
    missing_values = [MissingValue(time_stamp_utc=x) for x in dates]
    fc_result.ts_characteristics.outliers = outliers
    fc_result.ts_characteristics.change_points = change_points
    fc_result.ts_characteristics.missing_values = missing_values

    # Act
    characteristics = _prepare_characteristics(prepared_actuals=df, result=fc_result)

    # Assert
    assert characteristics.outliers == []
    assert characteristics.change_points == []
    assert characteristics.missing_values == []

    assert fc_result.ts_characteristics.outliers == outliers
    assert fc_result.ts_characteristics.change_points == change_points
    assert fc_result.ts_characteristics.missing_values == missing_values


def test__prepare_characteristics___all_value_are_within_timeframe___all_values_are_in_result(sample_fc_result_1):
    # Arrange
    df = create_example_df('monthly')
    fc_result = sample_fc_result_1
    dates = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
    outliers = [Outlier(time_stamp_utc=x, original_value=1) for x in dates]
    change_points = [ChangePoint(time_stamp_utc=x, change_point_type='LEVEL_SHIFT') for x in dates]
    missing_values = [MissingValue(time_stamp_utc=x) for x in dates]
    fc_result.ts_characteristics.outliers = outliers
    fc_result.ts_characteristics.change_points = change_points
    fc_result.ts_characteristics.missing_values = missing_values

    # Act
    characteristics = _prepare_characteristics(prepared_actuals=df, result=fc_result)

    # Assert
    assert characteristics.outliers == outliers
    assert characteristics.change_points == change_points
    assert characteristics.missing_values == missing_values

    assert fc_result.ts_characteristics.outliers == outliers
    assert fc_result.ts_characteristics.change_points == change_points
    assert fc_result.ts_characteristics.missing_values == missing_values


def test__prepare_characteristics___some_value_are_within_timeframe___expected_values_are_in_result(sample_fc_result_1):
    # Arrange
    df = create_example_df('monthly')
    fc_result = sample_fc_result_1
    excluded_date = datetime(2000, 1, 1)
    included_date = datetime(2020, 1, 1)
    dates = [included_date, excluded_date]
    outliers = [Outlier(time_stamp_utc=x, original_value=1) for x in dates]
    change_points = [ChangePoint(time_stamp_utc=x, change_point_type='LEVEL_SHIFT') for x in dates]
    missing_values = [MissingValue(time_stamp_utc=x) for x in dates]
    fc_result.ts_characteristics.outliers = outliers
    fc_result.ts_characteristics.change_points = change_points
    fc_result.ts_characteristics.missing_values = missing_values

    # Act
    characteristics = _prepare_characteristics(prepared_actuals=df, result=fc_result)

    # Assert
    assert excluded_date not in [x.time_stamp_utc for x in characteristics.outliers]
    assert included_date in [x.time_stamp_utc for x in characteristics.outliers]

    assert excluded_date not in [x.time_stamp_utc for x in characteristics.change_points]
    assert included_date in [x.time_stamp_utc for x in characteristics.change_points]

    assert excluded_date not in [x.time_stamp_utc for x in characteristics.missing_values]
    assert included_date in [x.time_stamp_utc for x in characteristics.missing_values]

    assert fc_result.ts_characteristics.outliers == outliers
    assert fc_result.ts_characteristics.change_points == change_points
    assert fc_result.ts_characteristics.missing_values == missing_values


def create_backtesting_value(date: datetime, fc_step: int, fc_value: float) -> BacktestingValue:
    """Helper to create a BacktestingValue for testing."""
    return BacktestingValue(
        time_stamp_utc=date,
        point_forecast_value=fc_value,
        lower_limit_value=fc_value - 1,
        upper_limit_value=fc_value + 1,
        fc_step=ValidatedPositiveInt(fc_step)
    )


def create_model_with_backtesting(backtesting_values: list[BacktestingValue]) -> Model:
    """Helper to create a Model with backtesting data."""
    return Model(
        model_name='TestModel',
        status=ModelStatus.Successful,
        forecast_plausibility=Plausibility('Plausible'),
        forecasts=[ForecastValue(
            time_stamp_utc=datetime(2023, 6, 1),
            point_forecast_value=100.0,
            lower_limit_value=90.0,
            upper_limit_value=110.0
        )],
        raw_model_forecasts=[ForecastValue(
            time_stamp_utc=datetime(2023, 6, 1),
            point_forecast_value=100.0,
            lower_limit_value=90.0,
            upper_limit_value=110.0
        )],
        model_selection=ComparisonDetails(
            backtesting=backtesting_values,
            plausibility=None,
            accuracy=[],
            ranking=RankingDetails(rank_position=PositiveInt(1), score=1.0)
        ),
        test_period=None
    )


def test__prepare_backtesting___valid_iteration___returns_expected_dataframe():
    # Arrange
    # Create backtesting with 3 fc_steps and 2 iterations each
    backtesting_values = [
        create_backtesting_value(date=datetime(2023, 1, 1), fc_value=1, fc_step=1),
        create_backtesting_value(date=datetime(2023, 2, 1), fc_value=1, fc_step=2),
        create_backtesting_value(date=datetime(2023, 3, 1), fc_value=1, fc_step=3),
        create_backtesting_value(date=datetime(2023, 2, 1), fc_value=2, fc_step=1),
        create_backtesting_value(date=datetime(2023, 3, 1), fc_value=2, fc_step=2),
        create_backtesting_value(date=datetime(2023, 4, 1), fc_value=2, fc_step=3)
    ]
    model = create_model_with_backtesting(backtesting_values)

    # Act
    result = _prepare_backtesting(model, iteration=1)

    # Assert
    assert len(result) == 3
    assert list(result.columns) == ['date', 'fc', 'lower', 'upper']
    assert result['date'].tolist() == [datetime(2023, 1, 1), datetime(2023, 2, 1), datetime(2023, 3, 1)]
    assert result['fc'].tolist() == [1, 1, 1]


def test__prepare_backtesting___iteration_too_high___raises_value_error():
    # Arrange
    backtesting_values = [
        create_backtesting_value(date=datetime(2023, 1, 1), fc_value=1, fc_step=1),
        create_backtesting_value(date=datetime(2023, 2, 1), fc_value=1, fc_step=2),
    ]
    model = create_model_with_backtesting(backtesting_values)

    # Act & Assert
    with pytest.raises(ValueError, match='Selected iteration was not calculated.'):
        _prepare_backtesting(model, iteration=2)


def test__prepare_backtesting___single_fc_step_multiple_iterations___returns_expected_dataframe():
    # Arrange
    backtesting_values = [
        create_backtesting_value(date=datetime(2023, 1, 1), fc_value=1, fc_step=1),
        create_backtesting_value(date=datetime(2023, 2, 1), fc_value=2, fc_step=1),
        create_backtesting_value(date=datetime(2023, 3, 1), fc_value=3, fc_step=1),
    ]
    model = create_model_with_backtesting(backtesting_values)

    # Act
    result = _prepare_backtesting(model, iteration=2)

    # Assert
    assert len(result) == 1
    assert result['fc'].tolist() == [2]
