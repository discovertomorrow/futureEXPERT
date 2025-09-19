"""Contains the models with the configuration for CHECK-IN."""
from datetime import datetime
from typing import Literal, Optional

import pydantic
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from futureexpert.shared_models import TimeSeries


class BaseConfig(BaseModel):
    """Basic Configuaration for all models."""
    model_config = ConfigDict(extra='forbid')


class FileSpecification(BaseConfig):
    """Specify the format of the CSV file.

    Parameters
    ----------
    delimiter
        The delimiter used to separate values.
    decimal
        The decimal character used in decimal numbers.
    thousands
        The thousands separator used in numbers.
    """
    delimiter: Optional[str] = ','
    decimal: Optional[str] = '.'
    thousands: Optional[str] = None


class Column(BaseConfig):
    """Base model for the different column models
    (`DateColumn`, `ValueColumn` and `GroupColumn`).

    Parameters
    ----------
    name
        The original name of the column.
    name_new
        The new name of the column.
    """
    name: str
    name_new: Optional[str] = None


class DateColumn(Column):
    """Model for the date columns.

    Parameters
    ----------
    format
        The format of the date.
    """
    format: str


class ValueColumn(Column):
    """Model for the value columns.

    Parameters
    ----------
    min
        The set minimum value of the column.
    max
        The set maximum value of the column.
    dtype_str
        The data type of the column.
    unit
        The unit of the column.
    """
    min: Optional[int] = None
    max: Optional[int] = None
    dtype_str: Optional[Literal['Numeric', 'Integer']] = None
    unit: Optional[str] = None


class GroupColumn(Column):
    """Model for the group columns.

    Parameters
    ----------
    dtype_str
        The data type of the column.
    """
    dtype_str: Optional[Literal['Character']] = None


class DataDefinition(BaseConfig):
    """Model for the input parameter needed for the first CHECK-IN step.
    Every single column in your data must be accounted for. Each column must either be assigned a type (`date_column`,
    `value_columns`, `group_columns`) or be explicitly marked for removal in `remove_columns`.

    Parameters
    ----------
    date_column
        Definition of the date column. Must be a single column that contains the complete date information.
    value_columns
        Definitions of the value columns. Not all columns defined here must be used for time series creation;
        selecting a subset or combining is possible in a later step.
    group_columns
        Definitions of the group columns. Not all columns defined here must be used for time series creation; selecting
        a subset is possible in a later step. Grouping information can also be used to create hierarchical levels.
    remove_rows
        Indexes of the rows to be removed before validation. Note: If the raw data was committed as pandas data frame
        the header is the first row (row index 0).
    remove_columns
        Indexes of the columns to be removed before validation. Any column that is not assigned a type must be listed here.
    """
    date_column: DateColumn
    value_columns: list[ValueColumn]
    group_columns: list[GroupColumn] = []
    remove_rows: Optional[list[int]] = []
    remove_columns: Optional[list[int]] = []


class FilterSettings(BaseConfig):
    """Model for the filters.

    Parameters
    ----------
    type
        The type of filter: `exclusion` or `inclusion`.
    variable
        The columns name to be used for filtering.
    items
        The list of values to be used for filtering.
    """
    type: Literal['exclusion', 'inclusion']
    variable: str
    items: list[str]


class NewValue(BaseConfig):
    """Model for the value data.

    Parameters
    ----------
    first_variable
        The first variable name.
    operator
        The operator that will be used to do the math operation
        between the first and second variable.
    second_variable
        The second variable name.
    new_variable
        The new variable name.
    unit
        The unit.
    """
    first_variable: str
    operator: Literal['x', '+', '-']
    second_variable: str
    new_variable: str
    unit: Optional[str] = None


class TsCreationConfig(BaseConfig):
    """Configuration for the creation of time series.

    Parameters
    ----------
    value_columns_to_save
        Value columns that should be saved.
    time_granularity
        Target granularity of the time series.
    description
        A short description of the time series.
    start_date
        Dates before this date are excluded.
    end_date
        Dates after this date are excluded.
    grouping_level
        Names of group columns that should be used as the grouping level.
    save_hierarchy
        If true, interpretes the given grouping levels as levels of a hierarchy and saves all hierachy levels.
        Otherwise, no hierarchy levels are implied and only the single level with the given grouping is saved.
        e.g. if grouping_level is ['A', 'B', 'C'] time series of grouping 'A', 'AB' and 'ABC' is saved.
        For later filtering use {'grouping.A': {'$exists': True}}
    filter
        Settings for including or excluding values during time series creation.
    new_variables
        New value column that is a combination of two other value columns.
    missing_value_handler
        Strategy how to handle missing values during time series creation.
    """
    value_columns_to_save: list[str]
    time_granularity: Literal['yearly', 'quarterly', 'monthly', 'weekly', 'daily', 'hourly', 'halfhourly']
    description: Optional[str] = None
    grouping_level: list[str] = []
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    save_hierarchy: bool = False
    filter: list[FilterSettings] = []
    new_variables: list[NewValue] = []
    missing_value_handler: Literal['keepNaN', 'setToZero'] = 'keepNaN'


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


class TimeSeriesVersion(BaseModel):
    """Time series version created in CHECK-IN.

    Parameters
    ----------
    version_id
        Id of the time series version. Used to identifiy the time series.
    description
        Description of the time series version.
    creation_time_utc
        Time of the creation.
    keep_until_utc: datetime
        Last day where the data is stored until it is deleted.
    """
    version_id: str
    description: Optional[str]
    creation_time_utc: datetime
    keep_until_utc: datetime

    @pydantic.model_validator(mode="after")
    def fix_time_stamps(self) -> Self:
        last_day = self.keep_until_utc.date()
        self.creation_time_utc = self.creation_time_utc.replace(microsecond=0)
        self.keep_until_utc = datetime.combine(last_day, datetime.min.time())

        return self
