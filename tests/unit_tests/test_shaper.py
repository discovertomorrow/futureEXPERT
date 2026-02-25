import math
from datetime import datetime

import pytest

from futureexpert.shaper import Scenario
from futureexpert.shared_models import Covariate, TimeSeries, TimeSeriesValue


def _create_scenario(num_values: int = 3) -> Scenario:
    """Helper to create a Scenario with a given number of high/low values."""
    timestamps = [datetime(2026, 1, i + 1) for i in range(num_values)]
    high = [TimeSeriesValue(time_stamp_utc=ts, value=float(i + 10)) for i, ts in enumerate(timestamps)]
    low = [TimeSeriesValue(time_stamp_utc=ts, value=float(i)) for i, ts in enumerate(timestamps)]
    return Scenario(
        ts=Covariate(ts=TimeSeries(name='temperature',
                                   group='test',
                                   granularity='daily',
                                   values=[TimeSeriesValue(time_stamp_utc=datetime(2026, 1, 1), value=3)]),
                     lag=0),
        ts_version='abc123',
        high=high,
        low=low,
    )


def test_add_custom_values___given_matching_length___sets_custom_values() -> None:

    # Arrange
    scenario = _create_scenario(num_values=3)
    custom_values = [5.0, 6.0, 7.0]

    # Act
    scenario.add_custom_values(custom_values)

    # Assert
    assert scenario.custom is not None
    assert len(scenario.custom) == 3
    for i, entry in enumerate(scenario.custom):
        assert entry.value == custom_values[i]
        assert entry.time_stamp_utc == scenario.high[i].time_stamp_utc


def test_add_custom_values___given_wrong_length___raises_value_error() -> None:

    # Arrange
    scenario = _create_scenario(num_values=3)

    # Act & Assert
    with pytest.raises(ValueError, match='All Scenarios need the same length'):
        scenario.add_custom_values([1.0, 2.0])


def test_add_custom_values___given_too_many_values___raises_value_error() -> None:

    # Arrange
    scenario = _create_scenario(num_values=3)

    # Act & Assert
    with pytest.raises(ValueError, match='All Scenarios need the same length'):
        scenario.add_custom_values([1.0, 2.0, 3.0, 4.0])


def test_add_custom_values___called_twice___overwrites_previous_custom() -> None:

    # Arrange
    scenario = _create_scenario(num_values=2)

    # Act
    scenario.add_custom_values([1.0, 2.0])
    scenario.add_custom_values([10.0, 20.0])

    # Assert
    assert scenario.custom is not None
    assert math.isclose(scenario.custom[0].value, 10.0)
    assert math.isclose(scenario.custom[1].value, 20.0)
