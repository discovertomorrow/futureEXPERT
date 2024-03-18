"""Contains the models with the configuration for the checkin."""
import json
from typing import Any, Literal, Optional

from pydantic import BaseModel


class FileSpecification(BaseModel):
    """Specify the format of the csv file.

    Parameters
    ----------
    delimiter
        The delimiter used to separate values.
    decimal
        The decimal character that is used in numbers.
    thousands
        The thousands character that is used in numbers.
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
    nameNew
        The new name of the column.
    """
    name: str
    nameNew: Optional[str] = None


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
    dtypeStr
        The data type of the column.
    unit
        The unit of the column.
    """
    min: Optional[int] = None
    max: Optional[int] = None
    dtypeStr: Optional[Literal['Numeric', 'Integer']] = None
    unit: Optional[str] = None


class GroupColumn(Column):
    """Model for the group columns.

    Parameters
    ----------
    dtypeStr
        The data type of the column.
    """
    dtypeStr: Optional[Literal['Character']] = None


class DataDefinition(BaseModel):
    """Model for the input parameter needed for the first checkin step.

    Parameters
    ----------
    remove_rows
        index of the rows that get deleted before any validation
    remove_columns
        index of the columns that get deleted before any validation
    date_columns
        definition of the date column
    value_columns
        definitions of the value columns
    group_columns
        definitions of the group columns
    """
    remove_rows: Optional[list[int]] = []
    remove_columns: Optional[list[int]] = []
    date_columns: DateColumn
    value_columns: list[ValueColumn]
    group_columns: list[GroupColumn] = []


def create_checkin_payload_1(user_input_id: str, file_uuid: str, data_definition: DataDefinition, file_specification: FileSpecification = FileSpecification()) -> Any:
    """Creates the payload for the checkin stage prepareDataset.

    Parameters
    ----------
    user_input_id
        UUID of the user input
    file_uuid
        UUID of the file
    data_definition
        Defines Column and row and column removal.
    file_specification
        Specify the format of the csv file. Only relevant if a csv was given as input.
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
                    'dateColumns': [data_definition.date_columns.model_dump(exclude_none=True)],
                    'valueColumns': [d.model_dump(exclude_none=True) for d in data_definition.value_columns],
                    'groupColumns': [d.model_dump(exclude_none=True) for d in data_definition.group_columns]
                }
            }}


def build_payload_from_ui_config(user_input_id: str, file_uuid: str, path: str) -> Any:
    """Creates the payload for the checkin stage createDataset.

    Parameters
    ----------
    user_input_id
        UUID of the user input
    file_uuid
        UUID of the file
    path
        path to the json file
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
        The type of filter. `exclusion` or `inclusion`
    variable
        The columns name that will be used for filtering.
    items
        The list of values that will be used for filtering.
    """
    type: Literal['exclusion', 'inclusion']
    variable: str
    items: list[str]


class NewValue(BaseModel):
    """Model for the value data.

    Parameters
    ----------
    firstVariable
        The first variable name.
    operator
        The operator that will be used to do the math operation
        between the first and second variable.
    secondVariable
        The second variable name.
    newVariable
        The new variable name.
    unit
        The unit.        
    """
    firstVariable: str
    operator: Literal['x', '+', '-']
    secondVariable: str
    newVariable: str
    unit: Optional[str] = None


class TsCreationConfig(BaseModel):
    """Model for the Time Series creation config.

    Parameters
    ----------
    timeGranularity
        Target granularity of the time series.
    startDate
        Dates before this date are cut off.
    endDate
        Dates after this date are cut off.
    grouppinglevel
        Name of group columns that should be used as grouping level.
    newVariables
        New value column that is a combination of two other value columns.
    valueColumnsToSave
        Value columns that should be saved.
    missingValueHandler
        Strategy how to handle missing values during time series creation.            
    """
    timeGranularity: Literal['yearly', 'monthly', 'weekly', 'daily']
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    grouppinglevel: list[str] = []
    filter: list[FilterSettings] = []
    newVariables: list[NewValue] = []
    valueColumnsToSave: list[str] = []
    missingValueHandler: Literal['keepNaN', 'setToZero'] = 'keepNaN'


def create_checkin_payload_2(payload: dict[str, Any], config: TsCreationConfig) -> Any:
    """Creates the payload for the checkin stage createDataset.

    Parameters
    ----------
    payload
        payload used in create_checkin_payload_1
    config
        Configuration for time series creation.
    """

    payload['payload']['rawDataReviewResults'] = {}
    payload['payload']['timeSeriesDatasetParameter'] = {
        'aggregation': {'operator': 'sum',
                        'option': config.missingValueHandler},
        'date': {
            'timeGranularity': config.timeGranularity,
            'startDate': config.startDate,
            'endDate': config.endDate
        },
        'grouping': {
            'dataLevel': config.grouppinglevel,
            'filter':  [d.model_dump() for d in config.filter]
        },
        'values': [d.model_dump() for d in config.newVariables],
        'valueColumnsToSave': config.valueColumnsToSave
    }
    payload['payload']['stage'] = 'createDataset'

    return payload
