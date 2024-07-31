"""Shared models used across multiple modules."""
from __future__ import annotations

from typing import Annotated

import pydantic


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
