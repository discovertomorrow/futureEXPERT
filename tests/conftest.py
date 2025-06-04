from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from futureexpert import ExpertClient
from futureexpert.forecast import (ComparisonDetails,
                                   ForecastInput,
                                   ForecastResult,
                                   ForecastValue,
                                   Model,
                                   ModelStatus,
                                   Plausibility,
                                   RankingDetails,
                                   TimeSeriesCharacteristics)
from futureexpert.shared_models import (Covariate,
                                        CovariateRef,
                                        PositiveInt,
                                        TimeSeries,
                                        TimeSeriesValue,
                                        ValidatedPositiveInt)


def create_forecast_result(actuals_name: str,
                           model_name: str,
                           season_length: int,
                           ts_class: str,
                           quantization: int) -> ForecastResult:

    # Helper function to create TimeSeries with example values
    def get_example_ts(name: str) -> TimeSeries:
        return TimeSeries(grouping={'level': 'a'},
                          granularity='monthly',
                          group='test-group',
                          name=name,
                          values=[TimeSeriesValue(time_stamp_utc=datetime(2000, 1, 1), value=1.0)])

    actuals = get_example_ts(actuals_name)
    cov_name = 'cov'
    covs = get_example_ts(cov_name)

    ts_characteristics = TimeSeriesCharacteristics(
        season_length=[ValidatedPositiveInt(season_length)],
        ts_class=ts_class,
        quantization=quantization
    )

    models = [Model(
        model_name=model_name,
        status=ModelStatus.Successful,
        forecast_plausibility=Plausibility('Plausible'),
        forecasts=[ForecastValue(lower_limit_value=0,
                                 upper_limit_value=0,
                                 point_forecast_value=0,
                                 time_stamp_utc=datetime(2020, 2, 1)),
                   ForecastValue(lower_limit_value=0,
                                 upper_limit_value=0,
                                 point_forecast_value=0,
                                 time_stamp_utc=datetime(2020, 3, 1)),
                   ForecastValue(lower_limit_value=0,
                                 upper_limit_value=0,
                                 point_forecast_value=0,
                                 time_stamp_utc=datetime(2020, 4, 1))],
        model_selection=ComparisonDetails(
            backtesting=[],
            plausibility=None,
            accuracy=[],
            ranking=RankingDetails(rank_position=PositiveInt(1), score=1.0)
        ),
        covariates=[CovariateRef(lag=0, name=cov_name)],
        test_period=None
    )]

    forecast_result = ForecastResult(
        input=ForecastInput(
            actuals=actuals,
            covariates=[Covariate(lag=0, ts=covs)]
        ),
        ts_characteristics=ts_characteristics,
        models=models,
        changed_start_date=None,
        changed_values=[]
    )

    return forecast_result


@pytest.fixture()
def sample_fc_result_1() -> ForecastResult:
    return create_forecast_result(
        actuals_name='actuals_1',
        model_name='ARIMA',
        season_length=6,
        ts_class='smooth',
        quantization=2
    )


@pytest.fixture()
def sample_fc_result_2() -> ForecastResult:
    return create_forecast_result(
        actuals_name='actuals_2',
        model_name='CART',
        season_length=12,
        ts_class='erratic',
        quantization=3
    )


@pytest.fixture()
def sample_fc_result_3() -> ForecastResult:
    return create_forecast_result(
        actuals_name='actuals_3',
        model_name='TBATS',
        season_length=4,
        ts_class='lumpy',
        quantization=1
    )


@pytest.fixture(scope="module")
def demand_planning_results() -> list[ForecastResult]:
    client = ExpertClient()
    all_reports = client.get_reports(limit=100)
    demand_planning_report = all_reports[all_reports["description"] ==
                                                     'Monthly Demand Forecast on Material Level'].iloc[0]
    return client.get_fc_results(id=demand_planning_report["report_id"], include_k_best_models=3)


@pytest.fixture(scope="module")
def sales_forecasting_result() -> list[ForecastResult]:
    client = ExpertClient()
    all_reports = client.get_reports(limit=100)
    sales_forecasting_report = all_reports[all_reports["description"] ==
                                                       'Monthly Sales Forecast on Country Level'].iloc[0]
    return client.get_fc_results(id=sales_forecasting_report["report_id"], include_k_best_models=3)
