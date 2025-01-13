"""Contains the models with the configuration for CHECK-IN."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

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

    Parameters
    ----------
    remove_rows
        Indexes of the rows to be removed before validation. Note: If the raw data was committed as pandas data frame
        the header is the first row (row index 0).
    remove_columns
        Indexes of the columns to be removed before validation.
    date_columns
        Definition of the date column.
    value_columns
        Definitions of the value columns.
    group_columns
        Definitions of the group columns.
    """
    remove_rows: Optional[list[int]] = []
    remove_columns: Optional[list[int]] = []
    date_columns: DateColumn
    value_columns: list[ValueColumn]
    group_columns: list[GroupColumn] = []


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
    """Model for the time series creation configuration.

    Parameters
    ----------
    time_granularity
        Target granularity of the time series.
    start_date
        Dates before this date are excluded.
    end_date
        Dates after this date are excluded.
    grouping_level
        Names of group columns that should be used as the grouping level.
    filter
        Settings for including or excluding values during time series creation.
    new_variables
        New value column that is a combination of two other value columns.
    value_columns_to_save
        Value columns that should be saved.
    missing_value_handler
        Strategy how to handle missing values during time series creation.
    """
    time_granularity: Literal['yearly', 'quarterly', 'monthly', 'weekly', 'daily', 'hourly', 'halfhourly']
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    grouping_level: list[str] = []
    filter: list[FilterSettings] = []
    new_variables: list[NewValue] = []
    value_columns_to_save: list[str]
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
