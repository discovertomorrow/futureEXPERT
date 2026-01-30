from datetime import datetime

import pytest
from notebook_execution.utility import extract_report_ids_from_notebook

from futureexpert import ExpertClient
from futureexpert.forecast import (ComparisonDetails,
                                   ForecastInput,
                                   ForecastResult,
                                   ForecastResults,
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

    forecast_values = [ForecastValue(lower_limit_value=0,
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
                                     time_stamp_utc=datetime(2020, 4, 1))]
    models = [Model(
        model_name=model_name,
        status=ModelStatus.Successful,
        forecast_plausibility=Plausibility('Plausible'),
        forecasts=forecast_values,
        raw_model_forecasts=forecast_values,
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


DEMAND_PLANNING_NOTEBOOK_PATHS = [
    './use_cases/demand_planning/demand_planning.ipynb',
    './use_cases/demand_planning/demand_planning_NOW+EXPERT.ipynb',
]


@pytest.fixture(params=DEMAND_PLANNING_NOTEBOOK_PATHS, scope='session')
def demand_planning_results(request: pytest.FixtureRequest) -> ForecastResults:
    [report_id] = extract_report_ids_from_notebook(path=request.param)
    client = ExpertClient()
    return client.get_fc_results(
        id=report_id,
        include_k_best_models=20,
        include_discarded_models=True
    )


@pytest.fixture(scope='session')
def demand_planning_reference_result() -> ForecastResults:
    reference_report_id = 150698
    reference_report_group = 'gitlab-ci-futureexpert'
    reference_report_environment = 'development'

    with ExpertClient(group=reference_report_group,
                      environment=reference_report_environment) as client:
        return client.get_fc_results(
            id=reference_report_id,
            include_k_best_models=20,
            include_discarded_models=True
        )


@pytest.fixture(scope='session')
def sales_forecasting_result() -> ForecastResults:
    notebook_path = './use_cases/sales_forecasting/sales_forecasting.ipynb'
    report_ids = extract_report_ids_from_notebook(path=notebook_path)
    client = ExpertClient()
    [report_id] = [report_id for report_id in report_ids if client.get_report_type(report_id) == 'forecast']
    return client.get_fc_results(
        id=report_id,
        include_k_best_models=3,
        include_discarded_models=True
    )


@pytest.fixture(scope='session')
def sales_forecasting_reference_result() -> ForecastResults:
    reference_report_id = 170547
    reference_report_group = 'gitlab-ci-futureexpert'
    reference_report_environment = 'development'

    with ExpertClient(group=reference_report_group,
                      environment=reference_report_environment) as client:
        return client.get_fc_results(
            id=reference_report_id,
            include_k_best_models=3,
            include_discarded_models=True
        )


@pytest.fixture(scope='module')
def expert_client() -> ExpertClient:
    return ExpertClient()
