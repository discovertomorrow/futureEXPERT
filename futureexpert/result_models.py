"""Models that are used for futureMATCHER and futureFORECAST results."""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat

from futureexpert.base_models import ValidatedPositiveInt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesValue(BaseModel):
    """Value of a time series.

    Parameters
    ----------
    time_stamp_utc
        The time stamp of the value.
    value
        The value.
    """
    time_stamp_utc: datetime
    value: Annotated[float, Field(allow_inf_nan=False)]


class TimeSeries(BaseModel):
    """Time series data.

    Parameters
    ----------
    name
        Name of the time series.
    group
        Group of the time series.
    granularity
        Granularity of the time series.
    values
        The actual values of the time series.
    """
    name: Annotated[str, Field(min_length=1)]
    group: str
    granularity: Annotated[str, Field(min_length=1)]
    values: Annotated[Sequence[TimeSeriesValue], Field(min_length=1)]


class CovariateRef(BaseModel):
    """Covariate reference.

    Parameters
    ----------
    name
        Name of the Covariate
    lag
        Lag by which the covariate was used.
    """
    name: str
    lag: int


class Covariate(BaseModel):
    """Covariate.

    Parameters
    ----------
    ts
        Time series object of the covariate. Not lagged.
    lag
        Lag by which the covariate was used.
    """
    ts: TimeSeries
    lag: int


class ModelStatus(str, Enum):
    """Supported status of a model."""
    Successful = 'Successful'
    FallbackSuccessful = 'FallbackSuccessful'
    InsufficientCovs = 'InsufficientCovs'
    RemovedByPreselection = 'RemovedByPreselection'
    Unknown = 'Unknown'
    NoEvaluationPossible = 'NoEvaluationPossible'
    NoForecastPossible = 'NoForecastPossible'
    NoTestPeriodEvaluationPossible = 'NoTestPeriodEvaluationPossible'
    MissingPredictionIntervals = 'MissingPredictionIntervals'


class Plausibility(str, Enum):
    """Supported plausibilities."""
    NotCalculated = 'NotCalculated'
    Plausible = 'Plausible'
    Implausible = 'Implausible'


class RankingDetails(BaseModel):
    """Details of ranking of a model.

    Parameters
    ----------
    rank_position
        The rank position of the model.
    score
        The score used to rank the model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    rank_position: ValidatedPositiveInt
    score: float


class ForecastValue(BaseModel):
    """Point forecast value with corridor of a forecast.

    Parameters
    ----------
    time_stamp_utc
        The time stamp of the forecast value.
    point_forecast_value
        The forecast value.
    lower_limit_value
        Lower limit of the prediction interval.
    upper_limit_value
        Upper limit of the prediction interval.
    """
    time_stamp_utc: datetime
    point_forecast_value: Annotated[float, Field(allow_inf_nan=False)]
    lower_limit_value: Optional[Annotated[float, Field(allow_inf_nan=False)]]
    upper_limit_value: Optional[Annotated[float, Field(allow_inf_nan=False)]]


class BacktestingValue(ForecastValue):
    """Point forecast value with corridor of a forecast step in backtesting."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    fc_step: ValidatedPositiveInt


class AccuracyMeasurement(BaseModel):
    """Accuracy measurement of a specific model and measure.

    Parameters
    ----------
    measure_name
        Name of the accuracy measure.
    value
        Value of the accuracy measure
    index
        Fc step or backtesting iteration used for calculating the accuracy measure
    aggregation_method
        Method used to compute measure values from BT results.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    measure_name: Annotated[str, Field(min_length=1)]
    value: float
    index: ValidatedPositiveInt
    aggregation_method: Annotated[str, Field(min_length=1)]


class ComparisonDetails(BaseModel):
    """Provides details about backtesting, ranking and accuracy measures.

    Parameters
    ----------
    backtesting
        The forecasts used for model evaluation.
    accuracy
        Calculated accuracy measures.
    ranking
        Details about the ranking of the models.
    plausibility
        Plausibility status of the backtesting.
    """
    backtesting: Sequence[BacktestingValue]
    accuracy: Sequence[AccuracyMeasurement]
    ranking: Optional[RankingDetails]
    plausibility: Optional[Plausibility]


class Model(BaseModel):
    """Model which was created for a specific run.

    Parameters
    ----------
    model_name
        Name of the model.
    status
        Status of the model.
    forecast_plausibility
        Plausibility status of the forecast.
    forecasts
        Forecast values from the model.
    model_selection
        Details about the model selection.
    test_period
        Details about the test period (if calculated).
    covariates
        Information about the covariate if one was used.
    method_specific_details
        Some additional method specific information.
    """
    model_name: Annotated[str, Field(min_length=1)]
    status: ModelStatus
    forecast_plausibility: Plausibility
    forecasts: Sequence[ForecastValue]
    model_selection: ComparisonDetails
    test_period: Optional[ComparisonDetails]
    covariates: Sequence[Union[CovariateRef]] = Field(default_factory=list)
    method_specific_details: Any = None

    model_config = ConfigDict(
        protected_namespaces=()  # ignore warnings about field names starting with 'model_'
    )


class Trend(BaseModel):
    """Trend details.

    Parameters
    ----------
    trend_probability
        The probability of the detected trend.
    is_trending
        Indicates whether a trend has been detected for the time series
    """
    trend_probability: Annotated[float, Field(ge=0., le=1., allow_inf_nan=False)]
    is_trending: bool


class ChangePoint(BaseModel):
    """Details about change points of the time series.

    Parameters
    ----------
    time_stamp_utc
        The time stamp with a detected change point.
    change_point_type
        The type of the change point
    """
    time_stamp_utc: datetime
    change_point_type: str


class Outlier(BaseModel):
    """Details about an outlier of the time series.

    Parameters
    ----------
    time_stamp_utc
        The time stamp with a detected outlier.
    original_value
        The value of the detected outlier.
    """
    time_stamp_utc: datetime
    original_value: float


class MissingValue(BaseModel):
    """Details about a missing value of the time series.

    Parameters
    ----------
    time_stamp_utc
        The time stamp with a missing value.
    """
    time_stamp_utc: datetime


class ChangedValue(BaseModel):
    """Details about a changed value of the time series.

    Parameters
    ----------
    time_stamp_utc
        The time stamp of the value.
    changed_value
        The replacement value.
    original_value
        The original value.
    change_reason
        The reason why a value was changed.
    """
    time_stamp_utc: datetime
    changed_value: float
    original_value: float
    change_reason: str


class ChangedStartDate(BaseModel):
    """Details about a changed start date of the time series.

    Parameters
    ----------
    original_start_date
        The original start date.
    changed_start_date
        The used start date.
    change_reason
        The reason why the start date was changed.
    """
    original_start_date: datetime
    changed_start_date: datetime
    change_reason: str


class TimeSeriesCharacteristics(BaseModel):
    """Characteristics of a time series.

    Parameters
    ----------
    season_lag
        Season lag of the time series.
    ts_class
        The time series class.
    trend
        Details about the trend.
    share_of_zeros
        Share of the series values that are zero.
    mean_inter_demand_interval
        The mean number of zeros between non-zero observations in the series.
    squared_coefficient_of_variation
        Squared coefficient of variation of the series values.
    quantization
        Detected value of the basic quantity.
    change_points
        Details about detected change points.
    outliers
        Details about detected outliers.
    missing_values
         Details about missing values.
    num_trailing_zeros
        Number of trailing zero values
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    season_lag: Optional[ValidatedPositiveInt] = None
    ts_class: Optional[Annotated[str, Field(min_length=1)]] = None
    trend: Optional[Trend] = None
    recent_trend: Optional[Trend] = None
    share_of_zeros: Optional[Annotated[float, Field(ge=0., le=1.)]] = None
    mean_inter_demand_interval: Optional[NonNegativeFloat] = None
    squared_coefficient_of_variation: Optional[NonNegativeFloat] = None
    quantization: Optional[float] = None
    change_points: Optional[Sequence[ChangePoint]] = None
    outliers: Optional[Sequence[Outlier]] = None
    missing_values: Optional[Sequence[MissingValue]] = None
    num_trailing_zeros: Optional[int] = None


class CovariateRankingDetails(BaseModel):
    """Final rank for a given set of covariates.

    Parameters
    ----------
    rank
        Rank for the given set of covariates.
    covariates
        Used covariates (might be zero or more than one).
    ts_ids
        Ts Ids of the used covariates.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rank: ValidatedPositiveInt
    covariates: list[Covariate]


class ForecastInput(BaseModel):
    """Input data of a forecast run.

    Parameters
    ----------
    actuals
        Time series for which the forecasts where performed.
    covariates
        Covariates if used.
    """
    actuals: TimeSeries
    covariates: Sequence[Covariate]


class ForecastResult(BaseModel):
    """Result of a forecast run and the corresponding input data.

    Parameters
    ----------
    input
        Actuals and Covariates of the forecast run.
    ts_characteristics
        Characteristics of the time series.
    changed_start_date
        Details about a changed start date of the time series.
    changed_values
        Details about changed value of the time series.
    models
        Details about all models that where ranked.
    """
    input: ForecastInput
    ts_characteristics: TimeSeriesCharacteristics
    changed_start_date: Optional[ChangedStartDate] = None
    changed_values: Sequence[ChangedValue]
    models: list[Model]

    def incorporate_matcher_ranking(self, matcher_results: list[MatcherResult]) -> ForecastResult:
        """Combines the ranking with the ranking from the matcher run.

        Parameters
        ----------
        matcher_results
            Results of a covariate matcher run and the corresponding input data.

         Returns
        -------
        The forecast results with the models adjusted based on the matcher ranking.
        """
        fc_result = deepcopy(self)
        actuals_name = self.input.actuals.name
        matcher_rankings = [item.ranking for item in matcher_results if item.actuals.name == actuals_name]

        if len(matcher_rankings) > 1:
            raise ValueError('Invalid matcher results found: Only one result per time series is permitted.')
        if len(matcher_rankings) == 0:
            logger.info(f'For {actuals_name} no MATCHER results were found. FORECAST ranking is used instead.')
            return fc_result

        new_models = []
        for matcher_ranking_details in matcher_rankings[0]:
            covs_config = [CovariateRef(name=cov.ts.name, lag=cov.lag) for cov in matcher_ranking_details.covariates]

            valid_models = [model for model in fc_result.models
                            if model.covariates == covs_config]

            valid_models.sort(key=lambda model: model.model_selection.ranking.rank_position)  # type: ignore

            if valid_models:
                selected_model = valid_models[0]
                selected_model.model_selection.ranking.rank_position = matcher_ranking_details.rank  # type: ignore
                new_models.append(selected_model)

        if len(new_models) != len(matcher_rankings[0]):
            logging.info('''Some MATCHER models are missing in the FORECAST. This could be caused by\n
                            - the model failed in the forecast run\n
                            - not all forecast models were downloaded
                            (for more model results, adjust parameter include_k_best_models)''')
        new_models.sort(key=lambda model: model.model_selection.ranking.rank_position)  # type: ignore
        fc_result.models = new_models
        return fc_result


class MatcherResult(BaseModel):
    """Result of a covariate matcher run and the corresponding input data.

    Parameters
    ----------
    actuals
        Time series for which the matching was performed.
    ranking
        Ranking of the different covariate and non-covariate models.
    """
    actuals: TimeSeries
    ranking: list[CovariateRankingDetails]

    def convert_ranking_to_forecast_config(self) -> ActualsCovsConfiguration:
        """Converts MATCHER results into the input format for the FORECAST."""
        covs_config = [CovariateRef(name=cov.ts.name, lag=cov.lag) for r in self.ranking for cov in r.covariates]
        return ActualsCovsConfiguration(actuals_name=self.actuals.name,
                                        covs_configurations=covs_config)


class ActualsCovsConfiguration(BaseModel):
    """Configuration of actuals and covariates via name and lag."""
    actuals_name: str
    covs_configurations: list[CovariateRef]


class CheckInResult(BaseModel):
    """Result of the CHECK-IN.

    Parameters
    ----------
    time_series
        Time series values.
    version_id
        Id of the time series version. Used to identifiy the time series
    """
    time_series: list[TimeSeries]
    version_id: str


def combine_forecast_ranking_with_matcher_ranking(forecast_results: list[ForecastResult],
                                                  matcher_results: list[MatcherResult]) -> list[ForecastResult]:
    """Ranks the forecasts based on the matcher results.

    Parameters
    ----------
    forecast_results
        Results of a forecast run and the corresponding input data.
    matcher_results
        Results of a covariate matcher run and the corresponding input data.

    Returns
    -------
    The forecast results with the models adjusted based on the matcher ranking.
    """
    new_results = []

    for fc_res in forecast_results:
        fc_res_new = fc_res.incorporate_matcher_ranking(matcher_results)
        new_results.append(fc_res_new)

    return new_results
