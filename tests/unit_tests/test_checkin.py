import numpy as np
import pandas as pd

import futureexpert.checkin as checkin
from futureexpert import DataDefinition, ExpertClient, TsCreationConfig


def test_CheckinConfig___given_minimum_input_parameter___runs_without_error() -> None:

    assert DataDefinition(date_column=checkin.DateColumn(name='test_a', format='%Y-%b-%d'),
                          value_columns=[checkin.ValueColumn(name='test_b')],
                          group_columns=[checkin.GroupColumn(name='test_c')])


def test_check_in_time_series___given_data_frame___runs_without_error(expert_client: ExpertClient) -> None:
    # Arrange
    values = np.arange(80)
    dates = pd.date_range("2018-01-01", periods=80, freq="d")
    raw_data = pd.DataFrame({'date': dates, 'value': values, 'type': 'success'})
    data_definition = DataDefinition(date_column=checkin.DateColumn(name='date', format='%Y-%m-%d'),
                                     value_columns=[checkin.ValueColumn(name='value')])
    config_ts_creation = TsCreationConfig(description='test_checkin',
                                          time_granularity='daily', value_columns_to_save=['value'])

    # Act
    expert_client.check_in_time_series(raw_data_source=raw_data,
                                       data_definition=data_definition,
                                       config_ts_creation=config_ts_creation)

    # Assert
    # no error
