"""Contains the models with the configuration for the forecast and the result format."""
from __future__ import annotations

import logging
import re
from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pydantic
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat
from typing_extensions import NotRequired, Self, TypedDict

from futureexpert.matcher import ActualsCovsConfiguration, MatcherResult
from futureexpert.pool import PoolCovDefinition
from futureexpert.shared_models import (BaseConfig,
                                        Covariate,
                                        CovariateRef,
                                        PositiveInt,
                                        TimeSeries,
                                        ValidatedPositiveInt)

logger = logging.getLogger(__name__)


class PreprocessingConfig(BaseConfig):
    """Preprocessing configuration.

    Parameters
    ----------
    remove_leading_zeros
        If true, then leading zeros are removed from the time series before forecasting. Is only applied
        if the time series has at least 5 values, including missing values.
    use_season_detection
        If true, then the season length is determined from the data.
    seasonalities_to_test
        Season lengths to be tested. If not defined, a suitable set for the given granularity is used.
        Season lengths can only be tested, if the number of observations is at least three times as
        long as the biggest season length. Note that 1 must be in the list if the non-seasonal case should
        be considered, too. Allows a combination of single granularities or combinations of granularities.
    fixed_seasonalities
        Season lengths used without checking. Allowed only if `use_season_detection` is false.
    detect_outliers
        If true, then identifies outliers in the data.
    replace_outliers
        If true, then identified outliers are replaced.
    detect_changepoints
        If true, then change points such as level shifts are identified.
    detect_quantization
        If true, a quantization algorithm is applied to the time series. Recognizes quantizations in the historic
        time series data and, if one has been detected, applies it to the forecasts.
    phase_out_method
        Choose which method will be used to detect Phase-Out in timeseries or turn it OFF.
        TRAILING_ZEROS method uses the number of trailing zeros to detect Phase-Out.
        AUTO_FEW_OBS method uses few-observation-changepoints at the end of the time series to detect Phase-Out.
        AUTO_FEW_OBS is only allowed if `detect_changepoints` is true.
    num_trailing_zeros_for_phase_out
        Number of trailing zeros in timeseries to detect Phase-Out with TRAILING_ZEROS method.
    recent_trend_num_observations
        Number of observations which are included in time span used for recent trend detection.
    recent_trend_num_seasons
        Number of seasons which are included in time span used for recent trend detection.
        If both recent_trend_num_seasons and recent_trend_num_observations are set, the longer time span is used.
    """

    remove_leading_zeros: bool = False
    use_season_detection: bool = True
    # empty lists and None are treated the same in apollon
    seasonalities_to_test: Optional[list[Union[list[ValidatedPositiveInt], ValidatedPositiveInt]]] = None
    fixed_seasonalities: Optional[list[ValidatedPositiveInt]] = None
    detect_outliers: bool = False
    replace_outliers: bool = False
    detect_changepoints: bool = False
    detect_quantization: bool = False
    phase_out_method: Literal['OFF', 'TRAILING_ZEROS', 'AUTO_FEW_OBS'] = 'OFF'
    num_trailing_zeros_for_phase_out: ValidatedPositiveInt = PositiveInt(5)
    recent_trend_num_observations: Optional[ValidatedPositiveInt] = PositiveInt(6)
    recent_trend_num_seasons: Optional[ValidatedPositiveInt] = PositiveInt(2)

    @pydantic.model_validator(mode='after')
    def _has_no_fixed_seasonalities_if_uses_season_detection(self) -> Self:
        if self.use_season_detection and self.fixed_seasonalities:
            raise ValueError('If fixed seasonalities is enabled, then season detection must be off.')

        return self

    @pydantic.model_validator(mode='after')
    def _has_detect_changepoints_if_phase_out_method_is_auto_few_obs(self) -> Self:
        if not self.detect_changepoints and self.phase_out_method == 'AUTO_FEW_OBS':
            raise ValueError('If phase_out_method is set to AUTO_FEW_OBS, then detect_changepoints must be on.')

        return self

    @pydantic.model_validator(mode='after')
    def _has_no_recent_trend_num_observation_nor_num_seasons(self) -> Self:
        if not self.recent_trend_num_observations and not self.recent_trend_num_seasons:
            raise ValueError(
                'Both recent_trend_num_observations and recent_trend_num_seasons cannot be None at the same time.')

        return self


class ForecastingConfig(BaseConfig):
    """Forecasting configuration.

    Parameters
    ----------
    fc_horizon
        Forecast horizon.
    lower_bound
        Lower bound applied to the time series and forecasts.
    upper_bound
        Upper bound applied to the time series and forecasts.
    confidence_level
        Confidence level for prediction intervals.
    round_forecast_to_integer
        If true, then forecasts are rounded to the nearest integer (also applied during backtesting).
    use_ensemble
        If true, then calculate ensemble forecasts. Automatically makes a smart decision on which
        methods to use based on their backtesting performance.
    """

    fc_horizon: Annotated[ValidatedPositiveInt, pydantic.Field(ge=1, le=60)]
    lower_bound: Union[float, None] = None
    upper_bound: Union[float, None] = None
    confidence_level: float = 0.75
    round_forecast_to_integer: bool = False
    use_ensemble: bool = False

    @property
    def numeric_bounds(self) -> tuple[float, float]:
        return (
            self.lower_bound if self.lower_bound is not None else -np.inf,
            self.upper_bound if self.upper_bound is not None else np.inf,
        )


ForecastingMethods = Literal['AdaBoost', 'Aft4Sporadic', 'ARIMA', 'AutoEsCov', 'CART',
                             'CatBoost', 'Croston', 'ES', 'ExtraTrees', 'FoundationModel', 'Glmnet(l1_ratio=1.0)',
                             'MA(granularity)', 'InterpolID', 'LightGBM', 'LinearRegression',
                             'MedianAS', 'MedianPattern', 'MLP', 'MostCommonValue', 'MA(3)',
                             'Naive', 'RandomForest', 'MA(season lag)', 'SVM', 'TBATS', 'Theta',
                             'TSB', 'XGBoost', 'ZeroForecast']

AdditionalCovMethod = Literal['AdaBoost', 'ARIMA', 'CART', 'CatBoost', 'ExtraTrees', 'FoundationModel',
                              'Glmnet(l1_ratio=1.0)', 'LightGBM', 'LinearRegression',
                              'MLP', 'RandomForest', 'SVM', 'XGBoost']


class MethodSelectionConfig(BaseConfig):
    """Method selection configuration.

    Parameters
    ----------
    number_iterations
        Number of backtesting iterations. At least 8 iterations are needed for empirical prediction intervals.
    shift_len
        Number of time points by which the test window is shifted between backtesting iterations.
    refit
        If true, then models are refitted for each backtesting iteration.
    default_error_metric
        Error metric applied to the backtesting error for non-sporadic time series.
    sporadic_error_metric
        Error metric applied to the backtesting errors for sporadic time series.
    additional_accuracy_measures
        Additional accuracy measures for solely reporting purposes.
        Does not affect internal evaluation or model ranking.
    step_weights
        Mapping from forecast steps to weights associated with forecast errors for the given forecasting step.
        Only positive weights are allowed. Leave a forecast step out to assign a zero weight.
        Used only for non-sporadic time series. If empty, all forecast steps are weighted equally.
    additional_cov_method
        Define up to one additional method that uses the defined covariates for creating forecasts. Will not be
        calculated if deemed unfit by the preselection. If the parameter forecasting_methods
        is defined, the additional cov method must appear in that list, too.
    cov_combination
        Create a forecast model for each individual covariate (single)
        or a model using all covariates together (joint).
    forecasting_methods
        Define specific forecasting methods to be tested for generating forecasts.
        Specifying fewer methods can significantly reduce the runtime of forecast creation.
        If not specified, all available forecasting methods will be used by default.
        Given methods are automatically preselected based on time series characteristics of your data.
        If none of the given methods fits your data, a fallback set of forecasting methods will be used instead.
    phase_out_fc_methods
        List of methods that will be used to forecast phase-out time series.
        Phase-out detection must be enabled in preprocessing configuration to take effect.
    """

    number_iterations: Annotated[ValidatedPositiveInt, pydantic.Field(ge=1, le=24)] = PositiveInt(12)
    shift_len: ValidatedPositiveInt = PositiveInt(1)
    refit: bool = False
    default_error_metric: Literal['me', 'mpe', 'mse', 'mae', 'mase', 'mape', 'smape'] = 'mse'
    sporadic_error_metric: Literal['pis', 'sapis', 'acr', 'mar', 'msr'] = 'pis'
    additional_accuracy_measures: list[Literal['me', 'mpe', 'mse', 'mae', 'mase', 'mape', 'smape', 'pis', 'sapis',
                                               'acr', 'mar', 'msr']] = pydantic.Field(default_factory=list)
    step_weights: Optional[dict[ValidatedPositiveInt, pydantic.PositiveFloat]] = None

    additional_cov_method: Optional[AdditionalCovMethod] = None
    cov_combination: Literal['single', 'joint'] = 'single'
    forecasting_methods: Sequence[ForecastingMethods] = pydantic.Field(default_factory=list)
    phase_out_fc_methods: Sequence[ForecastingMethods] = pydantic.Field(default_factory=lambda: ['ZeroForecast'])


class PipelineKwargs(TypedDict):
    preprocessing_config: PreprocessingConfig
    forecasting_config: ForecastingConfig
    method_selection_config: NotRequired[MethodSelectionConfig]


class ReportConfig(BaseConfig):
    """Forecast run configuration.

    Parameters
    ----------
    matcher_report_id
        Report ID of the covariate matcher.
    covs_versions
        List of versions of the covariates.
    covs_configuration
        Mapping from actuals and covariates. Use for custom covariate or adjusted matcher results.
        If the matcher results should be used without changes use `matcher_report_id` instead.
    title
        Title of the report.
    actuals_filter
        Filter criterion for actuals time series. The given actuals version is
        automatically added as additional filter criterion. Possible Filter criteria are all fields that are part
        of the TimeSeries class. e.g. {'name': 'Sales'}
        For more complex filter check: https://www.mongodb.com/docs/manual/reference/operator/query/#query-selectors
    max_ts_len
        At most this number of most recent observations is used. Check the variable MAX_TS_LEN_CONFIG
        for allowed configuration.
    preprocessing
        Preprocessing configuration.
    forecasting
        Forecasting configuration.
    method_selection
        Method selection configuration. If not supplied, then a granularity dependent default is used.
    pool_covs
        List of covariate definitions.
    db_name
        Only accessible for internal use. Name of the database to use for storing the results.
    priority
        Only accessible for internal use. Higher value indicate higher priority.
    """

    matcher_report_id: Optional[int] = None
    covs_versions: list[str] = Field(default_factory=list)
    covs_configuration: Optional[list[ActualsCovsConfiguration]] = None
    title: str
    actuals_filter: dict[str, Any] = Field(default_factory=dict)

    max_ts_len: Optional[int] = None

    preprocessing: PreprocessingConfig = PreprocessingConfig()
    forecasting: ForecastingConfig
    pool_covs: Optional[list[PoolCovDefinition]] = None
    method_selection: Optional[MethodSelectionConfig] = None
    db_name:  Optional[str] = None
    priority: Annotated[Optional[int], pydantic.Field(ge=0, le=10)] = None

    @pydantic.model_validator(mode="after")
    def _correctness_of_cov_configurations(self) -> Self:
        if (self.matcher_report_id or self.covs_configuration) and (
                len(self.covs_versions) == 0 and self.pool_covs is None):
            raise ValueError(
                'If one of `matcher_report_id` and `covs_configuration` is set also `covs_versions` needs to be set.')
        if (self.matcher_report_id is None and self.covs_configuration is None) and (
                self.covs_versions or self.pool_covs):
            raise ValueError(
                'If `covs_versions` or `pool_covs` is set ' +
                'either `matcher_report_id` or `covs_configuration` needs to be set.')
        if self.covs_configuration is not None and len(self.covs_configuration) == 0:
            raise ValueError('`covs_configuration` has length zero and therefore won`t have any effect. '
                             'Please remove the parameter or set to None.')
        return self

    @pydantic.model_validator(mode="after")
    def _only_one_covariate_definition(self) -> Self:
        fields = [
            'matcher_report_id',
            'pool_covs'
        ]

        set_fields = [field for field in fields if getattr(self, field) is not None]

        if len(set_fields) > 1:
            raise ValueError(f"Only one of {', '.join(fields)} can be set. Found: {', '.join(set_fields)}")

        return self

    @pydantic.model_validator(mode="after")
    def _backtesting_step_weights_refer_to_valid_forecast_steps(self) -> Self:
        if (self.method_selection
            and self.method_selection.step_weights
                and max(self.method_selection.step_weights.keys()) > self.forecasting.fc_horizon):
            raise ValueError('Step weights must not refer to forecast steps beyond the fc_horizon.')

        return self

    @pydantic.model_validator(mode="after")
    def _valid_covs_version(self) -> Self:
        for covs_version in self.covs_versions:
            if re.match('^[0-9a-f]{24}$', covs_version) is None:
                raise ValueError(f'Given covs_version "{covs_version}" is not a valid ObjectId.')
        return self

    @pydantic.model_validator(mode='after')
    def _has_valid_phase_out_detection_method_if_phase_out_fc_method_was_changed(self) -> Self:
        if ((self.method_selection and self.method_selection.phase_out_fc_methods != ['ZeroForecast']) and
                self.preprocessing.phase_out_method == 'OFF'):
            # A warning is logged instead of raising an error since this does not cause downstream issues.
            # The user is informed that their changes to phase_out_fc_methods have no effect
            # to clarify the relationship between these settings.
            logger.warning('Phase-out detection must be enabled in PreprocessingConfig'
                           ' so changes in phase_out_fc_methods in MethodSelectionConfig take effect.')
        return self

    @pydantic.model_validator(mode='after')
    def _has_non_empty_phase_out_fc_method_if_phase_out_detection_is_on(self) -> Self:
        if (self.method_selection and
                not self.method_selection.phase_out_fc_methods and
                self.preprocessing.phase_out_method != 'OFF'):
            raise ValueError('Phase out forecasting method cannot be empty when phase out detection is enabled.')

        return self


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
    score: Annotated[float, Field(allow_inf_nan=False)]


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
    original_value: Optional[Annotated[float, Field(allow_inf_nan=False)]]
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
    season_length
        Season length of the time series.
    ts_class
        The time series class.
    trend
        Details about the trend.
    recent_trend
        Details about the recent_trend. Is only detected if the time series has at least 10 values.
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
    season_length: Optional[Sequence[ValidatedPositiveInt]] = Field(alias="season_lag", default=None)
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


def export_result_overview_to_pandas(results: list[ForecastResult]) -> pd.DataFrame:
    """Extracts various time series insights, metadata, and other information from the forecast results
    and compiles them into an overview table. Contains model information about the best model.

    Parameters:
    -----------
    results
        The forecast results as provided by get_fc_results.

    Returns:
    --------
    A DataFrame where each row represents the insights for one time series.
    """

    overview_list = []
    for ts_result in results:
        [best_model] = [model for model in ts_result.models
                        if model.model_selection.ranking is not None
                        and model.model_selection.ranking.rank_position == 1]
        overview_for_ts: dict[str, Any] = {"name": ts_result.input.actuals.name}
        if ts_result.input.actuals.grouping:
            overview_for_ts.update(ts_result.input.actuals.grouping)
        overview_for_ts.update({
            'model': best_model.model_name,
            'cov': best_model.covariates[0].name if best_model.covariates else np.nan,
            'cov_lag': best_model.covariates[0].lag if best_model.covariates else np.nan,
            'season_length': ts_result.ts_characteristics.season_length,
            'ts_class': ts_result.ts_characteristics.ts_class,
            'quantization': (ts_result.ts_characteristics.quantization
                             if ts_result.ts_characteristics.quantization else np.nan),
            'trend': ts_result.ts_characteristics.trend.is_trending if ts_result.ts_characteristics.trend else np.nan,
            'recent_trend': (ts_result.ts_characteristics.recent_trend.is_trending
                             if ts_result.ts_characteristics.recent_trend else np.nan),
            'missing_values_count': (len(ts_result.ts_characteristics.missing_values)
                                     if ts_result.ts_characteristics.missing_values else np.nan),
            'outliers_count': (len(ts_result.ts_characteristics.outliers)
                               if ts_result.ts_characteristics.outliers else np.nan)
        })
        overview_list.append(overview_for_ts)
    return pd.DataFrame(overview_list)


def export_forecasts_to_pandas(results: list[ForecastResult]) -> pd.DataFrame:
    """Export forecasts of the best model including lower and upper limits of prediction interval.

    Parameters:
    -----------
    results
        The forecast results as provided by get_fc_results.

    Returns:
    --------
    A DataFrame where each row represents the forecast information of a single timeseries of a certain date.
    """
    forecast_list = []
    for ts_result in results:
        [best_model] = [model for model in ts_result.models
                        if model.model_selection.ranking is not None
                        and model.model_selection.ranking.rank_position == 1]
        for fc in best_model.forecasts:
            ts_forecast = {'name': ts_result.input.actuals.name}
            ts_forecast.update(fc)
            forecast_list.append(ts_forecast)
    return pd.DataFrame(forecast_list)


def export_forecasts_with_overview_to_pandas(results: list[ForecastResult]) -> pd.DataFrame:
    """Export forecasts with metadata from a list of ForecastResult objects to a pandas DataFrame.

    Parameters:
    -----------
    results
        The forecast results as provided by get_fc_results.

    Returns:
    --------
    A DataFrame where each row represents the forecast information of a single time series
    for a certain date, combined with the metadata.
    """

    metadata_df = export_result_overview_to_pandas(results)
    forecasts_df = export_forecasts_to_pandas(results)
    merged_df = pd.merge(forecasts_df, metadata_df, on='name', how='outer', validate="many_to_one")

    return merged_df
