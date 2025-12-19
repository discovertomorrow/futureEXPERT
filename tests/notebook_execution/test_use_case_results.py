import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from utility import extract_overview_table_from_notebook

from futureexpert.forecast import ForecastResults


def test_demand_planning_notebook___confirm_number_of_results(demand_planning_results: ForecastResults) -> None:
    assert len(demand_planning_results) == 16


def test_demand_planning_notebook___confirm_content_of_overview_table(demand_planning_results: ForecastResults) -> None:
    _assert_equal_overview_table_from_csv(
        demand_planning_results,
        'tests/notebook_execution/expected_demand_planning_overview.csv'
    )


def test_demand_planning_notebook___content_of_overview_table_equals_content_of_notebook(
    demand_planning_results: ForecastResults
) -> None:
    _assert_equal_overview_table_from_notebook(
        demand_planning_results,
        '././use_cases/demand_planning/demand_planning.ipynb'
    )


def test_demand_planning_notebook___confirm_aggregated_forecasts(
    demand_planning_results: ForecastResults,
    demand_planning_reference_result: ForecastResults
) -> None:
    _assert_equal_forecasts(demand_planning_results, demand_planning_reference_result)


# Sales Forecasting Tests

def test_sales_forecasting_notebook___confirm_number_of_results(sales_forecasting_result: ForecastResults) -> None:
    assert len(sales_forecasting_result) == 8


def test_sales_forecasting_notebook___content_of_overview_table_equals_content_of_notebook(
    sales_forecasting_result: ForecastResults
) -> None:
    _assert_equal_overview_table_from_notebook(
        sales_forecasting_result,
        '././use_cases/sales_forecasting/sales_forecasting.ipynb'
    )


def test_sales_forecasting_notebook___confirm_content_of_overview_table(
        sales_forecasting_result: ForecastResults
) -> None:
    _assert_equal_overview_table_from_csv(
        sales_forecasting_result,
        'tests/notebook_execution/expected_sales_forecasting_overview.csv'
    )


def test_sales_forecasting___confirm_aggregated_forecasts(
    sales_forecasting_result: ForecastResults,
    sales_forecasting_reference_result: ForecastResults
) -> None:
    _assert_equal_forecasts(sales_forecasting_result, sales_forecasting_reference_result)


def _assert_equal_values(values: list, ref_values: list, description: str, errors: list) -> None:
    try:
        assert_series_equal(pd.Series(values), pd.Series(ref_values), obj=description)
    except AssertionError as e:
        errors.append(f'{description}:\n{e}')


def _assert_equal_overview_table_from_csv(results: ForecastResults, csv_path: str) -> None:
    expected_overview = pd.read_csv(
        csv_path,
        sep=';',
        converters={'season_length': pd.eval}
    ).sort_values(by='name').reset_index(drop=True)
    result_df = results.export_result_overview_to_pandas().sort_values(by='name').reset_index(drop=True)
    assert_frame_equal(result_df, expected_overview)


def _assert_equal_overview_table_from_notebook(results: ForecastResults, notebook_path: str) -> None:
    expected_overview = extract_overview_table_from_notebook(notebook_path)
    result_df = results.export_result_overview_to_pandas().sort_values(by='name').reset_index(drop=True)
    assert_frame_equal(result_df, expected_overview)


def _assert_equal_forecasts(results: ForecastResults, reference_results: ForecastResults) -> None:
    assert len(results) == len(reference_results)
    errors = []

    for result in results:
        actuals_name = result.input.actuals.name
        reference_result = reference_results.get_forecast_result(actuals_name)

        if reference_result is None:
            errors.append(f'No reference result found for {actuals_name}')
            continue

        if len(result.models) != len(reference_result.models):
            errors.append(f'Different number of models for {actuals_name}: '
                          f'{len(result.models)} vs {len(reference_result.models)}')
            continue

        for model in result.models:
            ref_model = reference_result.get_model(name=model.model_name, covariates=model.covariates)

            if model.model_name != ref_model.model_name:
                errors.append(f'Model mismatch for {actuals_name}: '
                              f'{model.model_name} vs {ref_model.model_name}')
                continue

            if len(model.forecasts) != len(ref_model.forecasts):
                errors.append(f'Different number of forecasts for {actuals_name}, '
                              f'model {model.model_name}: '
                              f'{len(model.forecasts)} vs {len(ref_model.forecasts)}')
                continue

            comparisons = [
                ([f.time_stamp_utc for f in model.forecasts],
                 [f.time_stamp_utc for f in ref_model.forecasts],
                 f'Timestamps for {actuals_name}, model {model.model_name}'),
                ([f.point_forecast_value for f in model.forecasts],
                 [f.point_forecast_value for f in ref_model.forecasts],
                 f'Point forecasts for {actuals_name}, model {model.model_name}'),
            ]

            for values, ref_values, description in comparisons:
                _assert_equal_values(values, ref_values, description, errors)

    if errors:
        error_report = f'\n\n{'='*80}\nFound {len(errors)} assertion failure(s):\n{'='*80}\n\n'
        for i, error in enumerate(errors, 1):
            error_report += f'\n{i}. {error}\n{'-'*80}\n'
        pytest.fail(error_report)
