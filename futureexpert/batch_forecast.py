"""Contains the models with the configuration for the forecast."""
from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import pydantic
from typing_extensions import NotRequired, Self, TypedDict

from futureexpert.base_models import BaseConfig, PositiveInt, ValidatedPositiveInt
from futureexpert.result_models import ActualsCovsConfiguration


class PreprocessingConfig(BaseConfig):
    """

    Parameters
    ----------
    model_config
       Preprocessing configuration.
    remove_leading_zeros
       If true, then leading zeros are removed from the time series before forecasting.
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
       If true, then a quantization algorithm is applied to the time series.

    """

    # is merged with the inherited settings
    model_config = pydantic.ConfigDict(title='Preprocessing configuration.')

    remove_leading_zeros: bool = False
    use_season_detection: bool = True
    # empty lists and None are treated the same in apollon
    seasonalities_to_test: Optional[list[Union[list[ValidatedPositiveInt], ValidatedPositiveInt]]] = None
    fixed_seasonalities: Optional[list[ValidatedPositiveInt]] = None
    detect_outliers: bool = False
    replace_outliers: bool = False
    detect_changepoints: bool = False
    detect_quantization: bool = False

    @pydantic.model_validator(mode='after')
    def has_no_fixed_seasonalities_if_uses_season_detection(self) -> Self:
        if self.use_season_detection and self.fixed_seasonalities:
            raise ValueError('If fixed seasonalities is enabled, then season detection must be off.')

        return self


class ForecastingConfig(BaseConfig):
    """

    Parameters
    ----------
    model_config
       Preprocessing configuration.
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
       If true, then identified outliers are replaced.
    detect_changepoints
       If true, then calculate ensemble forecasts. Automatically makes a smart decision on which
       methods to use based on their backtesting performance.

    """

    # is merged with the inherited settings
    model_config = pydantic.ConfigDict(title='Forecasting configuration.')

    fc_horizon: ValidatedPositiveInt
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


class MethodSelectionConfig(BaseConfig):
    """

    Parameters
    ----------
    model_config
       Preprocessing configuration.
    number_iterations
       Number of backtesting iterations. At least 8 iterations are needed for empirical prediction intervals.
    shift_len
       Number of time points by which the test window is shifted between backtesting iterations.
    refit
       If true, then models are re-fitted for each backtesting iteration.
    default_error_metric
       Error metric applied to the backtesting error for non-sporadic time series.
    sporadic_error_metric
       Error metric applied to the backtesting errors for sporadic time series.
    step_weights
       Mapping from forecast steps to weights associated with forecast errors for the given forecasting step.
       Only positive weights are allowed. Leave a forecast step out to assign a zero weight.
       Used only for non-sporadic time series.
    detect_changepoints
       If true, then calculate ensemble forecasts. Automatically makes a smart decision on which
       methods to use based on their backtesting performance.

    """

    # is merged with the inherited settings
    model_config = pydantic.ConfigDict(title='Method selection configuration.')

    number_iterations: ValidatedPositiveInt = PositiveInt(12)
    shift_len: ValidatedPositiveInt = PositiveInt(1)
    refit: bool = False
    default_error_metric: Literal['me', 'mpe', 'mse', 'mae', 'mase', 'mape', 'smape'] = 'mse'
    sporadic_error_metric: Literal['pis', 'sapis', 'acr', 'mar', 'msr'] = 'pis'
    additional_accuracy_measures: list[Literal['me', 'mpe', 'mse', 'mae',
                                               'mase', 'mape', 'smape', 'pis', 'sapis', 'acr', 'mar', 'msr']] = []
    step_weights: Optional[dict[ValidatedPositiveInt, pydantic.PositiveFloat]] = None


class PipelineKwargs(TypedDict):
    preprocessing_config: PreprocessingConfig
    forecasting_config: ForecastingConfig
    method_selection_config: NotRequired[MethodSelectionConfig]


class ReportConfig(BaseConfig):
    """

    Parameters
    ----------
    model_config
       Preprocessing configuration.
    matcher_report_id
       Report ID of the covariate matcher.
    covs_version
       Version of the covariates.
    covs_configuration
       Mapping from actuals and covariates. Use for custom covariate or adjusted matcher results.
       If the matcher results should be used without changes use `matcher_report_id` instead.
    title
       Title of the report.
    max_ts_len
       At most this number of most recent observations is used. Check the environment variable MAX_TS_LEN_CONFIG
       for allowed configuration.
    preprocessing
       Preprocessing configuration.
    forecasting
       Forecasting configuration.
    backtesting
       Backtesting configuration. If not supplied, then a granularity dependent default is used
    db_name
       Only accessible for internal use. Name of the database to use for storing the results
    priority
       Only accessible for internal use. Higher value indicate higher priority
    """

    # is merged with the inherited settings
    model_config = pydantic.ConfigDict(title='Forecast run configuration.')
    matcher_report_id: Optional[int] = None
    covs_version: Optional[str] = None
    covs_configuration: Optional[list[ActualsCovsConfiguration]] = None
    title: str

    max_ts_len: Annotated[
        Optional[int], pydantic.Field(ge=1, le=1500)] = None

    preprocessing: PreprocessingConfig = PreprocessingConfig()
    forecasting: ForecastingConfig
    backtesting: Optional[MethodSelectionConfig] = None
    db_name:  Optional[str] = None
    priority: Annotated[Optional[int], pydantic.Field(ge=0, le=10)] = None

    @pydantic.model_validator(mode="after")
    def covs_configuration_not_with_matcher_report_id(self) -> Self:
        if self.matcher_report_id and self.covs_configuration:
            raise ValueError('matcher_report_id and covs_configuration can not be set simultaniusly.')
        if (self.matcher_report_id or self.covs_configuration) and self.covs_version is None:
            raise ValueError(
                'If one of `matcher_report_id` and `covs_configuration` is set also `covs_version` needs to be set.')
        if (self.matcher_report_id is None and self.covs_configuration is None) and self.covs_version:
            raise ValueError(
                'If `covs_version` is set either `matcher_report_id` or `covs_configuration` needs to be set.')
        if self.covs_configuration is not None and len(self.covs_configuration) == 0:
            raise ValueError('`covs_configuration` has length zero and therefore won`t have any effect.\
                             Please remove the parameter or set to None.')
        return self

    @pydantic.model_validator(mode="after")
    def backtesting_step_weights_refer_to_valid_forecast_steps(self) -> Self:
        if (self.backtesting
            and self.backtesting.step_weights
                and max(self.backtesting.step_weights.keys()) > self.forecasting.fc_horizon):
            raise ValueError('Step weights must not refer to forecast steps beyond the fc_horizon.')

        return self

    @pydantic.model_validator(mode="after")
    def valid_covs_version(self) -> Self:
        if self.covs_version is not None:
            if re.match('^[0-9a-f]{24}$', self.covs_version) is None:
                raise ValueError('Given covs_version is not a valid ObjectId.')

        return self

    @property
    def pipeline_kwargs(self) -> PipelineKwargs:
        pipeline_kwargs: PipelineKwargs = {
            'forecasting_config': self.forecasting,
            'preprocessing_config': self.preprocessing,
        }
        if self.backtesting is not None:
            pipeline_kwargs['method_selection_config'] = self.backtesting
        return pipeline_kwargs


def create_forecast_payload(version: str, config: ReportConfig) -> Any:
    """Creates the payload for the forecast.

    Parameters
    ----------
    version
      Version of the time series that should get forecasts.
    config
      Configuration of the forecast run.
    """

    config_dict = config.model_dump()
    config_dict['actuals_version'] = version
    config_dict['report_note'] = config_dict['title']
    config_dict['cov_selection_report_id'] = config_dict['matcher_report_id']
    config_dict['forecasting']['n_ahead'] = config_dict['forecasting']['fc_horizon']

    config_dict.pop('title')
    config_dict['forecasting'].pop('fc_horizon')
    config_dict.pop('matcher_report_id')

    payload = {'payload': config_dict}

    return payload


MAX_TS_LEN_CONFIG = {
    'halfhourly': {'default_len': 2*24*7,
                   'max_allowed_len': 1500},
    'hourly': {'default_len': 24*7*3,
               'max_allowed_len': 1500},
    'daily': {'default_len': 365,
              'max_allowed_len': 365*3},
    'weekly': {'default_len': 52*3,
               'max_allowed_len': 52*6},
    'monthly': {'default_len': 12*6,
                'max_allowed_len': 12*10},
    'quarterly': {'default_len': 4*12,
                  'max_allowed_len': 1500},
    'yearly': {'default_len': 1500,
               'max_allowed_len': 1500},
}


def calculate_max_ts_len(max_ts_len: Optional[int], granularity: str) -> Optional[int]:
    """Calculates the max_ts_len.

    Parameters
    ----------
    max_ts_len
        At most the number of most recent observations is used.
    granularity
       Granularity of the time series.
    """

    config = MAX_TS_LEN_CONFIG.get(granularity, None)
    assert config, 'For the given granularity no max_ts_len configuration exists.'
    default_len, max_allowed_len = config['default_len'], config['max_allowed_len']

    if max_ts_len is None:
        return default_len
    if max_ts_len > max_allowed_len:
        raise ValueError(
            f'''Given max_ts_len {max_ts_len} is not allowed for granularity {granularity}.
             Check the environment variable MAX_TS_LEN_CONFIG for allowed configuration.''')
    return max_ts_len


class MatcherConfig(BaseConfig):
    """Configuration for a futureMATCHER run.

    Parameters
    ----------
    title
        A short description of the report.
    actuals_version
        The version ID of the actuals.
    covs_version
        The version of the covariates.
    lag_selection_fixed_lags
        Lags that are tested in the lag selection.
    lag_selection_min_lag
        Minimal lag that is tested in the lag selection. For example, a lag 3 means the covariate
        is shifted 3 data points into the future.
    lag_selection_max_lag
        Maximal lag that is tested in the lag selection. For example, a lag 12 means the covariate
        is shifted 12 data points into the future.
    evaluation_start_date
        Optional start date for the evaluation. The input should be in the ISO format
        with date and time, "YYYY-mm-DDTHH-MM-SS", e.g., "2024-01-01T16:40:00".
        Actuals and covariate observations prior to this start date are dropped.
    evaluation_end_date
        Optional end date for the evaluation. The input should be in the ISO format
        with date and time, "YYYY-mm-DDTHH-MM-SS", e.g., "2024-01-01T16:40:00".
        Actuals and covariate observations after this end date are dropped.
    max_publication_lag
        Maximal publication lag for the covariates. The publication lag of a covariate
        is the number of most recent observations (compared to the actuals) that are
        missing for the covariate. E.g., if the actuals (for monthly granularity) end
        in April 2023 but the covariate ends in February 2023, the covariate has a
        publication lag of 2.
    post_selection_queries
        List of queries that are executed on the ranking summary DataFrame. Only ranking entries that
        match the queries are kept. The query strings need to satisfy the pandas query syntax
        (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html). Here are the columns
        of the ranking summary DataFrame that you might want to filter on:

        Column Name          |      Data Type   |    Description
        -----------------------------------------------------------------------------------------------
        Lag                  |          Int64   |    Lag of the covariate.
        Rank                 |        float64   |    Rank of the model.
        BetterThanNoCov      |           bool   |    Indicates whether the model is better than the non-cov model.
    enable_leading_covariate_selection
        When True, all covariates after the lag is applied that do not have at least one more
        datapoint beyond the the time period covered by actuals are removed from the candidate
        covariates passed to covariate selection.
    lag_selection_fixed_season_length
        An optional parameter specifying the length of a season in the dataset.
    """
    title: str
    actuals_version: str
    covs_version: str
    lag_selection_fixed_lags: Optional[list[int]] = None
    lag_selection_min_lag: Optional[int] = None
    lag_selection_max_lag: Optional[int] = None
    evaluation_start_date: Optional[str] = None
    evaluation_end_date: Optional[str] = None
    max_publication_lag: int = 2
    post_selection_queries: list[str] = []
    enable_leading_covariate_selection: bool = True
    lag_selection_fixed_season_length: Optional[int] = None

    @pydantic.model_validator(mode='after')
    def check_lag_selection_range(self) -> Self:

        min_lag = self.lag_selection_min_lag
        max_lag = self.lag_selection_max_lag

        if (min_lag is None) ^ (max_lag is None):
            raise ValueError(
                'If one of `lag_selection_min_lag` and `lag_selection_max_lag` is set the other one also needs to be set.')
        if min_lag and max_lag:
            if self.lag_selection_fixed_lags is not None:
                raise ValueError('Fixed lags and min/max lag are mutually exclusive.')
            if max_lag < min_lag:
                raise ValueError('lag_selection_max_lag needs to be higher as lag_selection_min_lag.')
            lag_range = abs(max_lag - min_lag)
            if lag_range > 15:
                raise ValueError(f'Only a range of 15 lags is allowed to test. The current range is {lag_range}.')

        return self

    @pydantic.model_validator(mode='after')
    def validate_post_selection_queries(self) -> Self:
        # Validate the post-selection queries.
        invalid_queries = []
        columns = {
            'Lag': 'int',
            'Rank': 'float',
            'BetterThanNoCov': 'bool'
        }
        # Create an empty DataFrame with the specified column names and data types
        validation_df = pd.DataFrame(columns=columns.keys()).astype(columns)
        for postselection_query in self.post_selection_queries:
            try:
                validation_df.query(postselection_query, )
            except Exception:
                invalid_queries.append(postselection_query)

        if len(invalid_queries):
            raise ValueError("The following post-selection queries are invalidly formatted: "
                             f"{', '.join(invalid_queries)}. ")

        return self


def create_matcher_payload(config: MatcherConfig) -> Any:

    config_dict: dict[str, Any] = {
        'report_description': config.title,
        'data_config': {
            'actuals_version': config.actuals_version,
            'actuals_filter': {},
            'covs_version': config.covs_version,
        },
        "compute_config": {
            "evaluation_start_date": config.evaluation_start_date,
            "evaluation_end_date": config.evaluation_end_date,
            "base_report_id": None,
            "base_report_requested_run_status": None,
            "report_update_strategy": "KEEP_OWN_RUNS",
            "cov_names": {
                'cov_name_prefix': '',
                'cov_name_field': 'name',
                'cov_name_suffix': '',
            },
            "preselection": {
                "min_num_actuals_obs": 78,
                "num_obs_short_term_class": 36,
                "max_publication_lag": config.max_publication_lag,
                "min_num_cov_obs": 96
            },
            "postselection": {
                "num_obs_short_term_correlation": 60,
                "clustering_run_id": None,
                "post_selection_queries": config.post_selection_queries,
                "post_selection_concatenation_operator": "&",
                "protected_selections_queries": [],
                "protected_selections_concatenation_operator": "&"
            },
            "lighthouse_config": {
                "enable_leading_covariate_selection": config.enable_leading_covariate_selection,
                "lag_selection_fixed_season_length": config.lag_selection_fixed_season_length,
                "lag_selection_fixed_lags": config.lag_selection_fixed_lags,
                "lag_selection_min_lag": config.lag_selection_min_lag,
                "lag_selection_max_lag": config.lag_selection_max_lag
            }
        }
    }

    return {'payload': config_dict}
