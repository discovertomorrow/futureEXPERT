import statistics

import pandas as pd
from numpy import isclose
from pandas.testing import assert_frame_equal
from utility import extract_overview_table_from_notebook

from futureexpert.forecast import ForecastResults


def test_demand_planning_notebook___confirm_number_of_results(demand_planning_results: ForecastResults) -> None:
    assert len(demand_planning_results) == 16


def test_demand_planning_notebook___confirm_content_of_overview_table(
        demand_planning_results: ForecastResults) -> None:
    # Arrange

    expected_overview = pd.read_csv("tests/notebook_execution/expected_demand_planning_overview.csv",
                                    sep=";", converters={'season_length': pd.eval}).sort_values(by="name").reset_index(drop=True)
    result_df = demand_planning_results.export_result_overview_to_pandas().sort_values(by="name").reset_index(drop=True)

    # Assert
    assert_frame_equal(result_df, expected_overview)


def test_demand_planning_notebook___content_of_overview_table_equals_content_of_notebook(
        demand_planning_results: ForecastResults) -> None:
    # Arrange
    path = '././use_cases/demand_planning/demand_planning.ipynb'
    expected_overview = extract_overview_table_from_notebook(path)
    result_df = demand_planning_results.export_result_overview_to_pandas().sort_values(by="name").reset_index(drop=True)

    # Assert
    assert_frame_equal(result_df, expected_overview)


def test_demand_planning_notebook___confirm_aggregated_forecasts(
        demand_planning_results: ForecastResults) -> None:
    # Arrange
    forecasts = [
        forecast.point_forecast_value
        for result in demand_planning_results
        for model in result.models
        for forecast in model.forecasts
    ]

    # Assert
    assert len(demand_planning_results) == 16
    assert isclose(sum(forecasts), 792930.0)
    assert isclose(statistics.stdev(forecasts), 2531.836)


def test_sales_forecasting_notebook___confirm_number_of_results(sales_forecasting_result: ForecastResults) -> None:
    assert len(sales_forecasting_result) == 8


def test_sales_forecasting_notebook___content_of_overview_table_equals_content_of_notebook(
        sales_forecasting_result: ForecastResults
) -> None:
    # Arrange
    path = '././use_cases/sales_forecasting/sales_forecasting.ipynb'
    expected_overview = extract_overview_table_from_notebook(path)
    result_df = sales_forecasting_result.export_result_overview_to_pandas().sort_values(by="name").reset_index(drop=True)

    # Assert
    assert_frame_equal(result_df, expected_overview)


def test_sales_forecasting_notebook___confirm_content_of_overview_table(
        sales_forecasting_result: ForecastResults
) -> None:
    # Arrange
    expected_overview = pd.read_csv(
        "tests/notebook_execution/expected_sales_forecasting_overview.csv",
        sep=";", converters={'season_length': pd.eval}
    ).sort_values(by="name").reset_index(drop=True)
    result_df = sales_forecasting_result.export_result_overview_to_pandas().sort_values(by="name").reset_index(drop=True)

    # Assert
    assert_frame_equal(result_df, expected_overview)


def test_sales_forecasting___confirm_aggregated_forecasts(
        sales_forecasting_result: ForecastResults
) -> None:
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
