import numpy as np
import pandas as pd
import pytest

from futureexpert.plot import _fill_missing_values_for_plot

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
