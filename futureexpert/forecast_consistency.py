"""Contains the models with the configuration for the hierarchical reconciliation and the result format."""
from __future__ import annotations

import logging
from typing import Optional

from pydantic import Field, model_validator

from futureexpert._forecast_consistency_metadata import (ConsistencyCheckResult,
                                                         ConsistentForecastMetadata,
                                                         ErrorSummaryStatistics,
                                                         FilteringSummary,
                                                         HierarchyMetadata,
                                                         PeriodSummary,
                                                         ReconciliationMethod,
                                                         ReconciliationSummary,
                                                         SeriesInconsistency,
                                                         ValidationResult)
from futureexpert.shared_models import BaseConfig

logger = logging.getLogger(__name__)


# Re-export all classes from _forecast_consistency_metadata for backward compatibility
__all__ = [
    # From _forecast_consistency_metadata
    'ReconciliationMethod',
    'HierarchyMetadata',
    'FilteringSummary',
    'PeriodSummary',
    'ReconciliationSummary',
    'ValidationResult',
    'ErrorSummaryStatistics',
    'SeriesInconsistency',
    'ConsistencyCheckResult',
    'ConsistentForecastMetadata',
    # Classes defined in this module
    'MakeForecastConsistentDataSelection',
    'ReconciliationConfig',
    'MakeForecastConsistentConfiguration',
]


class MakeForecastConsistentDataSelection(BaseConfig):
    """Forecast and time series selection for making forecast consistent.

    Parameters
    ----------
    version
        Time series version to be used.
    fc_report_id
        The identifier of the forecasting report to be used.
    forecast_minimum_version
        Optional version ID of time series containing minimum forecast values.
        Forecast minimums must match time series via grouping columns, granularity.
        Dates must be within the forecasting horizon.
    """
    version: str
    fc_report_id: int
    forecast_minimum_version: Optional[str] = None


class ReconciliationConfig(BaseConfig):
    """Configuration for hierarchical reconciliation process.

    Parameters
    ----------
    method
        Primary reconciliation method to use
    fallback_methods
        List of fallback methods to try if primary method fails
    excluded_levels
        Set of hierarchy levels to exclude from reconciliation
    actuals_period_length
        Number of last datapoints from actuals to use for proportion calculation (None = all)
    forecast_period_length
        Number of datapoints from forecasts to use for proportion calculation (None = all)
    round_forecast_to_integer
        If True, apply integer rounding constraint after reconciliation to ensure all
        forecast values are integers while preserving total sum and hierarchical consistency
    round_forecast_to_package_size
        If True, apply package size rounding constraint after reconciliation to ensure all
        forecast values are multiples of time series specific package sizes.
        Cannot be combined with round_forecast_to_integer.
    enforce_forecast_minimum_constraint
        If True, enforce forecast minimums from open orders or contractual obligations.
        Only available via client.start_making_forecast_consistent().
        Only available if round_forecast_to_package_size is active.
    """
    method: ReconciliationMethod = ReconciliationMethod.BOTTOM_UP
    fallback_methods: list[ReconciliationMethod] = Field(default_factory=list)
    excluded_levels: list[str] = Field(default_factory=list)
    actuals_period_length: Optional[int] = None
    forecast_period_length: Optional[int] = None
    round_forecast_to_integer: bool = False
    round_forecast_to_package_size: bool = False
    enforce_forecast_minimum_constraint: bool = False

    @model_validator(mode="after")
    def check_package_size_and_integer_rounding_exclusivity(self) -> ReconciliationConfig:
        """Validates that package size rounding and integer rounding cannot be used together."""
        if self.round_forecast_to_package_size and self.round_forecast_to_integer:
            raise ValueError(
                'round_forecast_to_package_size and round_forecast_to_integer cannot both be True. '
                'Package size rounding takes precedence and already enforces integer values.'
            )
        return self

    @model_validator(mode="after")
    def check_package_size_dependency(self) -> ReconciliationConfig:
        """Validates that package size rounding is active if minimum constraints are enforced."""
        if self.enforce_forecast_minimum_constraint and not self.round_forecast_to_package_size:
            raise ValueError(
                'enforce_forecast_minimum_constraint can only be True '
                'if round_forecast_to_package_size is also True.'
            )
        return self


class MakeForecastConsistentConfiguration(BaseConfig):
    """Service configuration.

    Parameters
    ----------
    data_selection
        Configuration on the selection of time series and forecasts used for carrying out the reconciliation.
    report_note
        Note of the report.
    db_name
        Only accessible for internal use. Name of the database to use for storing the results.
    reconciliation
        Optional reconciliation configuration. If not provided, defaults will be used.
    """
    data_selection: MakeForecastConsistentDataSelection
    report_note: str
    db_name: Optional[str] = None
    reconciliation: Optional[ReconciliationConfig] = None
