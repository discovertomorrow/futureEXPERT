import copy
import datetime as datetime

import futureexpert.forecast as fc
from futureexpert.matcher import CovariateRankingDetails, MatcherResult
from futureexpert.shared_models import Covariate, CovariateRef, TimeSeries, TimeSeriesValue


def test_combine_forecast_ranking_with_matcher_ranking___given_valid_inputs___runs_without_error() -> None:

    # Arrange
    actuals = TimeSeries(name='actuals', group='123', granularity='daily', values=[TimeSeriesValue(
        time_stamp_utc=datetime.datetime.strptime('2020-01-01', '%Y-%m-%d'), value=1)])

    list_covs = []
    ranking = [CovariateRankingDetails(rank=4, covariates=[])]

    for x in [1, 2, 3, 5]:
        name = f'cov{x}'
        cov = Covariate(ts=TimeSeries(name=name, group='123', granularity='daily', values=[
            TimeSeriesValue(time_stamp_utc=datetime.datetime.strptime('2020-01-01', '%Y-%m-%d'), value=1)]), lag=3)
        list_covs.append(cov)
        ranking.append(CovariateRankingDetails(rank=x, covariates=[cov]))

    matcher_results = MatcherResult(actuals=actuals, ranking=ranking)

    list_models = []
    for x in range(1, 8):
        model_name = f'mo{x}'
        covariate = []

        if x in [1, 2, 3, 4]:
            covariate.append(CovariateRef(name=list_covs[(x-1)].ts.name, lag=list_covs[(x-1)].lag))

        backtesting = [fc.BacktestingValue(time_stamp_utc=datetime.datetime.strptime('2020-01-01', '%Y-%m-%d'),
                                           point_forecast_value=1,
                                           lower_limit_value=None,
                                           upper_limit_value=None, fc_step=1)]
        model = fc.Model(model_name=model_name,
                         status='Successful',
                         forecast_plausibility='Plausible',
                         forecasts=[fc.ForecastValue(time_stamp_utc=datetime.datetime.strptime(
                             '2020-01-01', '%Y-%m-%d'), point_forecast_value=1, lower_limit_value=None, upper_limit_value=None)],
                         model_selection=fc.ComparisonDetails(backtesting=backtesting,
                                                              accuracy=[fc.AccuracyMeasurement(
                                                                  measure_name='test',
                                                                  value=1,
                                                                  index=1,
                                                                  aggregation_method='test')],
                                                              ranking=fc.RankingDetails(rank_position=8-x, score=x),
                                                              plausibility=None),
                         test_period=None,
                         covariates=covariate)
        list_models.append(model)
    forecast_result = fc.ForecastResult(input=fc.ForecastInput(actuals=actuals, covariates=list_covs),
                                        changed_values=[],
                                        ts_characteristics=fc.TimeSeriesCharacteristics(),
                                        models=list_models)
    # Forecast without matcher results
    actuals2 = copy.deepcopy(actuals)
    actuals2.name = "actuals with no matcher"
    forecast_result_2 = fc.ForecastResult(input=fc.ForecastInput(actuals=actuals2, covariates=[]),
                                          changed_values=[],
                                          ts_characteristics=fc.TimeSeriesCharacteristics(),
                                          models=list_models)

    # Act
    new_ranked_results = fc.combine_forecast_ranking_with_matcher_ranking(matcher_results=[matcher_results], forecast_results=[
        forecast_result, forecast_result_2])

    # Assert setup
    assert forecast_result.models[0].model_name == 'mo1'
    assert forecast_result.models[0].model_selection.ranking.rank_position == 7
    assert len(forecast_result.models) == 7

    # Assert result
    assert new_ranked_results[0].models[0].model_selection.ranking.rank_position == 1
    assert new_ranked_results[0].models[0].model_name == 'mo1'
    assert len(new_ranked_results[0].models) == 5
    assert len(new_ranked_results) == 2
