import futureexpert.checkin as checkin
from futureexpert import DataDefinition


def test_CheckinConfig___given_minimum_input_parameter___runs_without_error() -> None:

    assert DataDefinition(date_columns=checkin.DateColumn(name="test_a", format="%Y-%b-%d"),
                          value_columns=[checkin.ValueColumn(name="test_b")],
                          group_columns=[checkin.GroupColumn(name="test_c")])
