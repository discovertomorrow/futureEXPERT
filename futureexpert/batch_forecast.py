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


def _description(text: str) -> Any:
    return pydantic.Field(description=text)


class BaseConfig(pydantic.BaseModel):

    model_config = pydantic.ConfigDict(allow_inf_nan=False,
                                       extra='forbid',
                                       arbitrary_types_allowed=True)


class PreprocessingConfig(BaseConfig):

    # is merged with the inherited settings
    model_config = pydantic.ConfigDict(title='Preprocessing configuration.')

    remove_leading_zeros: Annotated[
        bool, _description('If true, then leading zeros are removed from the time series before forecasting.')
    ] = False
    use_season_detection: Annotated[
        bool, _description('If true, then the season length is determined from the data.')
    ] = False
    # empty lists and None are treated the same in apollon
    seasonalities_to_test: Annotated[
        Optional[list[ValidatedPositiveInt]],
        _description('Season lengths to be tested.'
                     ' If not defined, a suitable set for the given granularity is used.'
                     ' Note that 1 needs to be included here if the non-seasonal case should be considered, too.')
    ] = None
    fixed_seasonalities: Annotated[
        Optional[list[ValidatedPositiveInt]],
        _description('Season lengths used without checking. Allowed only if use_season_detection is false.')
    ] = None
    detect_outliers: Annotated[
        bool, _description('If true, then identifies outliers in the data.')
    ] = False
    replace_outliers: Annotated[
        bool, _description('If true, then identified outliers are replaced.')
    ] = False
    detect_changepoints: Annotated[
        bool, _description('If true, then change points such as level shifts are identified.')
    ] = False
    detect_quantization: Annotated[
        bool, _description('If true, then a quantization algorithm is applied to the time series.')
    ] = False

    @pydantic.model_validator(mode='after')
    def has_no_fixed_seasonalities_if_uses_season_detection(self) -> Self:
        if self.use_season_detection and self.fixed_seasonalities:
            raise ValueError('If fixed seasonalities should be used, then season detection must be off.')

        return self


class ForecastingConfig(BaseConfig):

    # is merged with the inherited settings
    model_config = pydantic.ConfigDict(title='Forecasting configuration.')

    fc_horizon: Annotated[ValidatedPositiveInt, _description('Forecast horizon.')]

    lower_bound: Annotated[
        Union[float, None], _description('Lower bound applied to the time series and forecasts.')
    ] = None
    upper_bound: Annotated[
        Union[float, None], _description('Upper bound applied to the time series and forecasts.')
    ] = None
    confidence_level: Annotated[
        float, pydantic.Field(gt=.0, lt=1., description='Confidence level for prediction intervals.')
    ] = 0.75
    round_forecast_to_integer: Annotated[
        bool, _description('If true, then forecasts are rounded to integer (also applied during backtesting).')
    ] = False

    @property
    def numeric_bounds(self) -> tuple[float, float]:
        return (
            self.lower_bound if self.lower_bound is not None else -np.inf,
            self.upper_bound if self.upper_bound is not None else np.inf,
        )


class MethodSelectionConfig(BaseConfig):

    # is merged with the inherited settings
    model_config = pydantic.ConfigDict(title='Method selection configuration.')

    number_iterations: Annotated[
        ValidatedPositiveInt, _description('Number of backtesting iterations.'
                                           ' At least 8 iterations are needed for empirical prediction intervals.')
    ] = PositiveInt(12)
    shift_len: Annotated[
        ValidatedPositiveInt,
        _description('Number of time points by which the test window is shifted between backtesting iterations.')
    ] = PositiveInt(1)
    refit: Annotated[bool, _description('If true, then models are re-fitted for each backtesting iteration.')] = False
    default_error_metric: Annotated[
        Literal['me', 'mpe', 'mse', 'mae', 'mase', 'mape', 'smape'],
        _description('Error metric applied to the backtesting error for non-sporadic time series.')
    ] = 'mse'
    sporadic_error_metric: Annotated[
        Literal['pis', 'sapis', 'acr', 'mar', 'msr'],
        _description('Error metric applied to the backtesting errors for sporadic time series.')
    ] = 'pis'
    step_weights: Annotated[
        Optional[dict[ValidatedPositiveInt, pydantic.PositiveFloat]],
        _description('Mapping from forecast steps to weights associated to forecast errors for that forecasting step.'
                     ' Only positive weights are allowed. Leave a forecast step out to assign a zero weight.'
                     ' Used only for non-sporadic time series.')
    ] = None


class PipelineKwargs(TypedDict):
    preprocessing_config: PreprocessingConfig
    forecasting_config: ForecastingConfig
    method_selection_config: NotRequired[MethodSelectionConfig]


class ReportConfig(BaseConfig):

    # is merged with the inherited settings
    model_config = pydantic.ConfigDict(title='Forecasting run configuration.')

    covs_version: Optional[str] = None
    covs_lag: Optional[int] = None
    title: str

    max_ts_len: Annotated[
        int, pydantic.Field(ge=1, description='At most this number of most recent observations is used.')
    ] = 84

    preprocessing: Annotated[PreprocessingConfig, _description("Preprocessing configuration.")] = PreprocessingConfig()
    forecasting: Annotated[ForecastingConfig, _description("Forecasting configuration.")]
    backtesting: Annotated[
        Optional[MethodSelectionConfig],
        _description('Backtesting configuration. If not supplied, then a granularity dependent default is used.')
    ] = None

    @pydantic.model_validator(mode="after")
    def backtesting_step_weights_refer_to_valid_forecast_steps(self) -> Self:
        if (self.backtesting
            and self.backtesting.step_weights
                and max(self.backtesting.step_weights.keys()) > self.forecasting.fc_horizon):
            raise ValueError('Step weights must not refer to forecast steps exceeding fc_horizon.')

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

    config
       Configuration of the forecast run.    
    """

    config_dict = config.model_dump()
    config_dict['actuals_version'] = version
    config_dict['report_note'] = config_dict['title']
    config_dict['forecasting']['n_ahead'] = config_dict['forecasting']['fc_horizon']

    config_dict.pop('title')
    config_dict['forecasting'].pop('fc_horizon')

    payload = {'payload': config_dict}

    return payload
