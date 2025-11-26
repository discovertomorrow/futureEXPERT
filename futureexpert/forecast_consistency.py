"""Contains the models with the configuration for the hierarchical reconciliation and the result format."""
from __future__ import annotations

import logging
from typing import Optional

from pydantic import Field

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
    """
    version: str
    fc_report_id: int


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
    """
    method: ReconciliationMethod = ReconciliationMethod.BOTTOM_UP
    fallback_methods: list[ReconciliationMethod] = Field(default_factory=list)
    excluded_levels: list[str] = Field(default_factory=list)
    actuals_period_length: Optional[int] = None
    forecast_period_length: Optional[int] = None


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
