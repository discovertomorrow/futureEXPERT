"""Contains the models with the configuration for the forecast."""
from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Optional, Union, cast

import numpy as np
import pydantic
from typing_extensions import NotRequired, Self, TypedDict


class PositiveInt(int):
    def __new__(cls, value: int) -> PositiveInt:
        if value < 1:
            raise ValueError('The value must be a positive integer.')
        return super().__new__(cls, value)


ValidatedPositiveInt = Annotated[PositiveInt,
                                 pydantic.BeforeValidator(lambda x: PositiveInt(int(x))),
                                 # raises an error without the lambda wrapper
                                 pydantic.PlainSerializer(lambda x: int(x), return_type=int),
                                 pydantic.WithJsonSchema({'type': 'int', 'minimum': 1})]


class BaseConfig(pydantic.BaseModel):

    model_config = pydantic.ConfigDict(allow_inf_nan=False,
                                       extra='forbid',
                                       arbitrary_types_allowed=True)


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
       Note that 1 must be in the list if the non-seasonal case should be considered, too.
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
    seasonalities_to_test: Optional[list[ValidatedPositiveInt]] = None
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
    covs_lag
       Lag for the covariate.
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
    covs_lag: Optional[int] = None
    title: str

    max_ts_len: Annotated[
        Optional[int], pydantic.Field(ge=1, le=1500)] = None

    preprocessing: PreprocessingConfig = PreprocessingConfig()
    forecasting: ForecastingConfig
    backtesting: Optional[MethodSelectionConfig] = None
    db_name:  Optional[str] = None
    priority: Annotated[Optional[int], pydantic.Field(ge=0, le=10)] = None

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
            f'Given max_ts_len {max_ts_len} is not allowed for granularity {granularity}. Check the environment variable MAX_TS_LEN_CONFIG for allowed configuration.')
    return max_ts_len


class MatcherConfig(BaseConfig):
    """Configuration for a futureMATCHER run.

    Parameters
    ----------
    title
      Title of the report.
    actuals_version
      Version of the 
    covs_version
      Version of the covariates.
    lag_selection_min_lag
      Minimal lag that is tested in the lag selection. For example, a lag 3 means the covariate is shifted 3 data points into the future.
    lag_selection_max_lag
      Maximal lag that is tested in the lag selection. For example, a lag 12 means the covariate is shifted 12 data points into the future.

    """
    title: str
    actuals_version: str
    covs_version: str
    lag_selection_min_lag: Optional[int] = None
    lag_selection_max_lag: Optional[int] = None

    @pydantic.model_validator(mode='after')
    def check_lag_selection_range(self) -> Self:

        min_lag = self.lag_selection_min_lag
        max_lag = self.lag_selection_max_lag

        if (min_lag is None) ^ (max_lag is None):
            raise ValueError(
                'If one of `lag_selection_min_lag` and `lag_selection_max_lag` is set the other one also needs to be set.')
        if min_lag and max_lag:
            if max_lag < min_lag:
                raise ValueError('lag_selection_max_lag needs to be higher as lag_selection_min_lag.')
            range = abs(max_lag - min_lag)
            if range > 15:
                raise ValueError(f'Only a range of 15 lags is allowed to test. The current range is {range}.')

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
            "evaluation_start_date": None,
            "evaluation_end_date": None,
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
                "max_publication_lag": 2,
                "min_num_cov_obs": 96
            },
            "postselection": {
                "num_obs_short_term_correlation": 60,
                "clustering_run_id": None,
                "post_selection_queries": [],
                "post_selection_concatenation_operator": "&",
                "protected_selections_queries": [],
                "protected_selections_concatenation_operator": "&"
            }
        }
    }

    if config.lag_selection_min_lag and config.lag_selection_max_lag:
        config_dict['compute_config']['lighthouse_config'] = {
            # The interchange and inversion of min_lag and max_lag is intended. Lags in Lighthouse configuration have the opposite meaning.
            'lag_selection_min_lag': -config.lag_selection_max_lag,
            'lag_selection_max_lag': -config.lag_selection_min_lag
        }

    payload = {'payload': config_dict}

    return payload
