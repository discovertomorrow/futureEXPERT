"""Shared models used across multiple modules."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Sequence

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, WithJsonSchema


class PositiveInt(int):
    def __new__(cls, value: int) -> PositiveInt:
        if value < 1:
            raise ValueError('The value must be a positive integer.')
        return super().__new__(cls, value)


ValidatedPositiveInt = Annotated[PositiveInt,
                                 BeforeValidator(lambda x: PositiveInt(int(x))),
                                 # raises an error without the lambda wrapper
                                 PlainSerializer(lambda x: int(x), return_type=int),
                                 WithJsonSchema({'type': 'int', 'minimum': 1})]


class BaseConfig(BaseModel):
    """Base configuration that is used for most models."""
    model_config = ConfigDict(allow_inf_nan=False,
                              extra='forbid',
                              arbitrary_types_allowed=True)


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