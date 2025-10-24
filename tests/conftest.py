from datetime import datetime
from typing import Any

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
from futureexpert.forecast_consistency import (ActualsMetadata,
                                               ConsistencyCheckResult,
                                               ConsistentForecast,
                                               ConsistentForecastResult,
                                               FilteringSummary,
                                               ForecastModel,
                                               HierarchyMetadata,
                                               OriginalForecast,
                                               OriginalForecastInput,
                                               PeriodSummary,
                                               ReconciliationMethod,
                                               ReconciliationSummary,
                                               ValidationResult)
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


DEMAND_PLANNING_REPORT_NAMES = [
    'Monthly Demand Forecast on Material Level NOW+EXPERT',
    'Monthly Demand Forecast on Material Level'
]


@pytest.fixture(scope='session')
def _all_demand_planning_results_cache():
    """Fetch results for all demand planning reports once and cache them."""

    client = ExpertClient()
    report_names = DEMAND_PLANNING_REPORT_NAMES
    all_reports = client.get_reports(limit=100)
    results_dict = {}
    for report_name in report_names:
        demand_planning_report = all_reports[all_reports['description'] == report_name].iloc[0]
        results = client.get_fc_results(
            id=demand_planning_report['report_id'],
            include_k_best_models=3
        )
        results_dict[report_name] = results
    return results_dict


@pytest.fixture(params=DEMAND_PLANNING_REPORT_NAMES)
def demand_planning_results(request, _all_demand_planning_results_cache):
    """Provide results for a single demand planning report, parametrized over all reports."""
    report_name = request.param
    return _all_demand_planning_results_cache[report_name]


@pytest.fixture(scope='module')
def sales_forecasting_result() -> list[ForecastResult]:
    client = ExpertClient()
    all_reports = client.get_reports(limit=100)
    sales_forecasting_report = all_reports[all_reports['description'] ==
                                           'Monthly Sales Forecast on Country Level'].iloc[0]
    return client.get_fc_results(id=sales_forecasting_report['report_id'], include_k_best_models=3)


@pytest.fixture(scope='module')
def expert_client() -> ExpertClient:
    return ExpertClient()


@pytest.fixture
def sample_hierarchical_forecasting_result() -> Any:
    """Complete HierarchicalForecastingResult with all fields populated."""
    name = 'Sales-North America'
    region = 'North America'
    forecast_values = [
        ForecastValue(
            time_stamp_utc=datetime(2024, 6, 1),
            point_forecast_value=110.0,
            lower_limit_value=105.0,
            upper_limit_value=115.0
        ),
        ForecastValue(
            time_stamp_utc=datetime(2024, 7, 1),
            point_forecast_value=115.0,
            lower_limit_value=110.0,
            upper_limit_value=120.0
        ),
        ForecastValue(
            time_stamp_utc=datetime(2024, 8, 1),
            point_forecast_value=120.0,
            lower_limit_value=115.0,
            upper_limit_value=125.0
        )
    ]

    original_forecast_1 = OriginalForecast(
        input=OriginalForecastInput(
            actuals=ActualsMetadata(
                name='Sales',
                grouping={}
            )
        ),
        models=[
            ForecastModel(
                model_name='TestModel',
                status='Successful',
                forecasts=forecast_values
            )
        ]
    )

    original_forecast_2 = OriginalForecast(
        input=OriginalForecastInput(
            actuals=ActualsMetadata(
                name=name,
                grouping={'Region': region}
            )
        ),
        models=[
            ForecastModel(
                model_name='TestModel',
                status='Successful',
                forecasts=[
                    ForecastValue(
                        time_stamp_utc=datetime(2024, 6, 1),
                        point_forecast_value=55.0,
                        lower_limit_value=50.0,
                        upper_limit_value=60.0
                    ),
                    ForecastValue(
                        time_stamp_utc=datetime(2024, 7, 1),
                        point_forecast_value=58.0,
                        lower_limit_value=53.0,
                        upper_limit_value=63.0
                    ),
                    ForecastValue(
                        time_stamp_utc=datetime(2024, 8, 1),
                        point_forecast_value=60.0,
                        lower_limit_value=55.0,
                        upper_limit_value=65.0
                    )
                ]
            )
        ]
    )

    reconciled_forecasts = [
        ConsistentForecast(
            ts_id='1078267',
            ts_name='Sales',
            original_forecast=original_forecast_1,
            forecasts=forecast_values
        ),
        ConsistentForecast(
            ts_id='1078268',
            ts_name=name,
            original_forecast=original_forecast_2,
            forecasts=[
                ForecastValue(
                    time_stamp_utc=datetime(2024, 6, 1),
                    point_forecast_value=55.0,
                    lower_limit_value=50.0,
                    upper_limit_value=60.0
                ),
                ForecastValue(
                    time_stamp_utc=datetime(2024, 7, 1),
                    point_forecast_value=58.0,
                    lower_limit_value=53.0,
                    upper_limit_value=63.0
                ),
                ForecastValue(
                    time_stamp_utc=datetime(2024, 8, 1),
                    point_forecast_value=60.0,
                    lower_limit_value=55.0,
                    upper_limit_value=65.0
                )
            ]
        )
    ]

    hierarchy_structure = HierarchyMetadata(
        total_levels=2,
        base_level=1,
        top_level=0,
        base_series_count=1,
        top_series_count=1,
        series_by_level={
            0: [{'ts_id': 1078267, 'name': 'Sales', 'grouping': {}}],
            1: [{'ts_id': 1078268, 'name': name, 'grouping': {'Region': region}}]
        },
        summing_matrix=None,
        tags={'Region': [region]}
    )

    filtering_summary = FilteringSummary(
        excluded_levels_applied=set(),
        original_series_count=2,
        filtered_series_count=0,
        remaining_series_count=2,
        filtered_ts_ids=[],
        filtered_by_level_count=0,
        filtering_applied=False
    )

    period_summary = PeriodSummary(
        final_actuals_length=15,
        final_forecasts_length=3,
        actuals_period_config=None,
        forecasts_period_config=None
    )

    reconciliation_summary = ReconciliationSummary(
        method_used=ReconciliationMethod.BOTTOM_UP,
        methods_attempted=[ReconciliationMethod.BOTTOM_UP],
        is_successful=True,
        total_series_reconciled=2,
        processing_time_seconds=0.15,
        error_message=None,
        quality_metrics={'mape': 2.5, 'rmse': 1.8}
    )

    validation_summary = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[]
    )

    consistency_check = ConsistencyCheckResult(
        total_aggregate_series=1,
        inconsistent_series_count=0,
        consistency_rate=100.0,
        overall_mape=0.0,
        individual_series_inconsistencies=[],
        summary_statistics={'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0},
        inconsistent_dates_ranked={}
    )

    return ConsistentForecastResult(
        reconciled_forecasts=reconciled_forecasts,
        reconciliation_method=ReconciliationMethod.BOTTOM_UP,
        model_selection_strategy='best_ranking_model',
        hierarchy_structure=hierarchy_structure,
        filtering_summary=filtering_summary,
        period_summary=period_summary,
        reconciliation_summary=reconciliation_summary,
        reconciliation_timestamp=datetime(2024, 9, 15, 10, 30, 0),
        validation_summary=validation_summary,
        consistency_check=consistency_check
    )
