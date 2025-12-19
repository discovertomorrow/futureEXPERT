"""Tests for ReconciliationConfig validators."""
import pytest

from futureexpert.forecast_consistency import ReconciliationConfig, ReconciliationMethod


def test_reconciliation_config___given_both_rounding_flags___raises_value_error() -> None:
    # Arrange
    expected_error_message = 'round_forecast_to_package_size and round_forecast_to_integer cannot both be True'

    # Assert
    with pytest.raises(ValueError, match=expected_error_message):
        ReconciliationConfig(
            method=ReconciliationMethod.BOTTOM_UP,
            round_forecast_to_package_size=True,
            round_forecast_to_integer=True
        )


def test_reconciliation_config___given_minimum_without_package_size___raises_value_error() -> None:
    # Arrange
    expected_error_message = 'enforce_forecast_minimum_constraint can only be True'

    # Assert
    with pytest.raises(ValueError, match=expected_error_message):
        ReconciliationConfig(
            method=ReconciliationMethod.BOTTOM_UP,
            enforce_forecast_minimum_constraint=True,
            round_forecast_to_package_size=False
        )
