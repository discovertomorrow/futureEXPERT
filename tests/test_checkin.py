from futureexpert import *


def test_CheckinConfig___given_minimum_input_parameter___runs_without_error() -> None:

    assert DataDefinition(date_columns=DateColumn(name="test_a", format="%Y-%b-%d"),
                          value_columns=[ValueColumn(name="test_b")],
                          group_columns=[GroupColumn(name="test_c")])
