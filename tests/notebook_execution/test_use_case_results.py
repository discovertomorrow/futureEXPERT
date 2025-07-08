import statistics

import pandas as pd
from numpy import isclose
from pandas.testing import assert_frame_equal

from futureexpert.forecast import export_result_overview_to_pandas


def test_demand_planning_notebook___confirm_number_of_results(demand_planning_results) -> None:
    assert len(demand_planning_results) == 16


def test_demand_planning_notebook___confirm_content_of_overview_table(
        demand_planning_results) -> None:
    # Arrange

    expected_overview = pd.read_csv("tests/notebook_execution/expected_demand_planning_overview.csv",
                                    sep=";", converters={'season_length': pd.eval}).sort_values(by="name").reset_index(drop=True)
    result_df = export_result_overview_to_pandas(demand_planning_results).sort_values(by="name").reset_index(drop=True)

    # Assert
    assert_frame_equal(result_df, expected_overview)


def test_demand_planning_notebook___confirm_aggregated_forecasts(
        demand_planning_results) -> None:
    # Arrange
    forecasts = [
        forecast.point_forecast_value
        for result in demand_planning_results
        for model in result.models
        for forecast in model.forecasts
    ]

    # Assert
    assert isclose(sum(forecasts), 792969.0)
    assert isclose(statistics.stdev(forecasts), 2531.823)


def test_sales_forecasting_notebook___confirm_number_of_results(sales_forecasting_result) -> None:
    assert len(sales_forecasting_result) == 8


def test_sales_forecasting_notebook___confirm_content_of_overview_table(
        sales_forecasting_result) -> None:
    # Arrange

    expected_overview = pd.read_csv("tests/notebook_execution/expected_sales_forecasting_overview.csv",
                                    sep=";", converters={'season_length': pd.eval}).sort_values(by="name").reset_index(drop=True)
    result_df = export_result_overview_to_pandas(sales_forecasting_result).sort_values(by="name").reset_index(drop=True)

    # Assert
    assert_frame_equal(result_df, expected_overview)


def test_sales_forecasting___confirm_aggregated_forecasts(
        sales_forecasting_result) -> None:
    # Arrange
    forecasts = [
        forecast.point_forecast_value
        for result in sales_forecasting_result
        for model in result.models
        for forecast in model.forecasts
    ]

    # Assert
    assert isclose(sum(forecasts), 585689.218)
    assert isclose(statistics.stdev(forecasts), 5185.352)
