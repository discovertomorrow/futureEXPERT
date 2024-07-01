"""Contains the models with the configuration for futureCHECK-IN."""
import json
from typing import Any, Literal, Optional

from pydantic import BaseModel


class FileSpecification(BaseModel):
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


class Column(BaseModel):
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


class DataDefinition(BaseModel):
    """Model for the input parameter needed for the first futureCHECK-IN step.

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


def create_checkin_payload_1(user_input_id: str, file_uuid: str, data_definition: DataDefinition, file_specification: FileSpecification = FileSpecification()) -> Any:
    """Creates the payload for the futureCHECK-IN stage prepareDataset.

    Parameters
    ----------
    user_input_id
        UUID of the user input.
    file_uuid
        UUID of the file.
    data_definition
        Specifies the data, value and group columns and which rows and columns are to be removed first.
    file_specification
        Specify the format of the CSV file. Only relevant if a CSV was given as input.
    """

    return {'userInputId': user_input_id,
            'payload': {
                'stage': 'prepareDataset',
                'fileUuid': file_uuid,
                'meta': file_specification.model_dump(),
                'performedTasks': {
                    'removedRows': data_definition.remove_rows,
                    'removedCols': data_definition.remove_columns
                },
                'columnDefinition': {
                    'dateColumns': [{snake_to_camel(key): value for key, value in data_definition.date_columns.model_dump(exclude_none=True).items()}],
                    'valueColumns': [{snake_to_camel(key): value for key, value in d.model_dump(exclude_none=True).items()} for d in data_definition.value_columns],
                    'groupColumns': [{snake_to_camel(key): value for key, value in d.model_dump(exclude_none=True).items()} for d in data_definition.group_columns]
                }
            }}


def build_payload_from_ui_config(user_input_id: str, file_uuid: str, path: str) -> Any:
    """Creates the payload for the futureCHECK-IN stage createDataset.

    Parameters
    ----------
    user_input_id
        UUID of the user input.
    file_uuid
        UUID of the file.
    path
        Path to the JSON file.
    """

    with open(path) as file:
        file_data = file.read()
        json_data = json.loads(file_data)

    json_data['stage'] = 'createDataset'
    json_data['fileUuid'] = file_uuid
    del json_data["performedTasksLog"]

    return {'userInputId': user_input_id,
            'payload': json_data}


class FilterSettings(BaseModel):
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


class NewValue(BaseModel):
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


class TsCreationConfig(BaseModel):
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
    time_granularity: Literal['yearly', 'monthly', 'weekly', 'daily', 'hourly']
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    grouping_level: list[str] = []
    filter: list[FilterSettings] = []
    new_variables: list[NewValue] = []
    value_columns_to_save: list[str] = []
    missing_value_handler: Literal['keepNaN', 'setToZero'] = 'keepNaN'


def create_checkin_payload_2(payload: dict[str, Any], config: TsCreationConfig) -> Any:
    """Creates the payload for the futureCHECK-IN stage createDataset.

    Parameters
    ----------
    payload
        Payload used in `create_checkin_payload_1`.
    config
        Configuration for time series creation.
    """

    payload['payload']['rawDataReviewResults'] = {}
    payload['payload']['timeSeriesDatasetParameter'] = {
        'aggregation': {'operator': 'sum',
                        'option': config.missing_value_handler},
        'date': {
            'timeGranularity': config.time_granularity,
            'startDate': config.start_date,
            'endDate': config.end_date
        },
        'grouping': {
            'dataLevel': config.grouping_level,
            'filter':  [d.model_dump() for d in config.filter]
        },
        'values': [{snake_to_camel(key): value for key, value in d.model_dump().items()} for d in config.new_variables],
        'valueColumnsToSave': config.value_columns_to_save
    }
    payload['payload']['stage'] = 'createDataset'

    return payload



def snake_to_camel(snake_string: str) -> str:
    """Converts snake case to lower camel case.

    Parameters
    ----------
    snake_string
        string im snake case format.
    """
    title_string = snake_string.title().replace('_', '')
    return title_string[:1].lower() + title_string[1:]
