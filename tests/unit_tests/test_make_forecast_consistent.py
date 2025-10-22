import pandas as pd

from futureexpert.make_forecast_consistent import export_consistent_forecasts_to_pandas


def test_export_consistent_forecasts_to_pandas(sample_hierarchical_forecasting_result):
    # Arrange
    results = sample_hierarchical_forecasting_result
    expected_columns = {'name', 'time_stamp_utc', 'point_forecast_value'}

    # Act
    df = export_consistent_forecasts_to_pandas(results)

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (6, 3)
    assert set(df.columns) == expected_columns
