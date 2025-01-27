from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from futureexpert.forecast import ChangePoint
from futureexpert.plot import _add_level_shifts, _calculate_few_observation_borders, _fill_missing_values_for_plot

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
        'value': rng.random(20)
    })


@pytest.mark.parametrize("granularity", granularities)
def test_fill_missing_values___no_missing_values___df_stays_unchanged(granularity):
    # Arrange
    df = create_example_df(granularity)
    # Act
    result_df = _fill_missing_values_for_plot(granularity, df)

    # Assert
    assert result_df.equals(df)


@pytest.mark.parametrize("granularity", granularities)
def test_fill_missing_values___with_missing_data___fills_missing_data_with_nan(granularity):
    # Arrange
    df = create_example_df(granularity)
    missing_indices = [2, 3, 7, 9, 11]
    df.drop(missing_indices, inplace=True)

    # Act
    result_df = _fill_missing_values_for_plot(granularity, df)

    # Assert
    assert result_df.shape[0] == 20, f"Expected 20 rows for {granularity}, but got {result_df.shape[0]}."
    assert result_df.loc[missing_indices, 'value'].isna().all()


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
    assert result[0][1] == datetime(2023, 2, 1)
    assert result[1][0] == datetime(2023, 2, 10)
    assert result[1][1] == datetime(2023, 2, 20)


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
    assert result[0][1] == datetime(2023, 3, 1)


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
    assert result[0][1] == datetime(2023, 3, 1)
    assert result[1][0] == datetime(2023, 4, 1)
    assert result[1][1] == datetime(2023, 4, 15)


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
