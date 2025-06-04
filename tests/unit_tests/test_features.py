from pathlib import Path

from dotenv import load_dotenv

import futureexpert.checkin as checkin
from futureexpert import DataDefinition, ExpertClient, FileSpecification, TsCreationConfig

load_dotenv()


def test_expert_client___given_save_hierarchy___returns_all_hierarchy() -> None:
    # Arrange
    client = ExpertClient()
    raw_data_source = Path("tests") / 'unit_tests/test_data/demo_demand_planning_data.csv'
    data_definition = DataDefinition(date_columns=checkin.DateColumn(name='DATE', format='%d.%m.%Y', name_new='Date'),
                                     value_columns=[checkin.ValueColumn(
                                         name='DEMAND', name_new='Demand')],
                                     group_columns=[checkin.GroupColumn(name="CUSTOMER", name_new='Customer'),
                                                    checkin.GroupColumn(name="MATERIAL", name_new="Material"),
                                                    checkin.GroupColumn(name="REGION", name_new="Region")])

    config_ts_creation = TsCreationConfig(time_granularity='monthly',
                                          start_date="2007-10-01",
                                          end_date="2024-06-01",
                                          value_columns_to_save=[
                                              'Demand'],
                                          grouping_level=[
                                              'Region', 'Customer'],
                                          save_hierarchy=True,
                                          missing_value_handler='setToZero')
    # Act
    check_in_version_id = client.check_in_time_series(raw_data_source=raw_data_source,
                                                      file_specification=FileSpecification(delimiter=';', decimal='.'),
                                                      data_definition=data_definition,
                                                      config_ts_creation=config_ts_creation)

    # Assert
    check_in_result = client.get_time_series(version_id=check_in_version_id)
    hierarchy_levels = list(set([tuple(ts.grouping.keys()) for ts in check_in_result.time_series]))
    assert len(check_in_result.time_series) == 9
    assert len(hierarchy_levels) == 3
    assert () in hierarchy_levels
    assert ('Region',) in hierarchy_levels
    assert ('Region', 'Customer') in hierarchy_levels or ('Customer', 'Region') in hierarchy_levels
