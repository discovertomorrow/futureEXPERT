"""Contains the models with the result metadata format for the hierarchical reconciliation.

Import all classes from futureexpert.forecast_consistency instead of using this internal module directly.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ReconciliationMethod(str, Enum):
    """Reconciliation methods for hierarchical forecasting."""

    BOTTOM_UP = 'bottom_up'
    """Sums forecasts from the base level of the hierarchy up to the top.

    Uses `hierarchicalforecast.methods.BottomUp`.
    """

    TOP_DOWN_PROPORTION_AVERAGES = 'top_down_proportion_averages'
    """Disaggregates the top-level forecast based on historical average proportions.

    Uses `hierarchicalforecast.methods.TopDown(method='proportion_averages')`.
    """

    TOP_DOWN_FORECAST_PROPORTION = 'top_down_forecast_proportion'
    """Disaggregates the top-level forecast based on the proportions of the base forecasts for each forecast step.

    Uses `hierarchicalforecast.methods.TopDown(method='forecast_proportions')`.
    """

    TOP_DOWN_AVERAGE_FORECAST_PROPORTION = 'top_down_average_forecast_proportion'
    """Disaggregates the top-level forecast based on the average proportions of the base forecasts over the horizon."""

    MIN_TRACE_WLS_STRUCT = 'min_trace_wls_struct'
    """Weights are based on the number of aggregated base series (Structural).

    Uses `hierarchicalforecast.methods.MinTrace(method='wls_struct')`.
    """


class HierarchyMetadata(BaseModel):
    """Metadata about hierarchical structure.

    Parameters
    ----------
    total_levels
        Total number of hierarchy levels detected
    base_level
        Index of the base (leaf) level in the hierarchy
    top_level
        Index of the top (root) level in the hierarchy
    base_series_count
        Number of time series at the base level
    top_series_count
        Number of time series at the top level
    series_by_level
        Dictionary mapping level indices to list of series metadata
    summing_matrix
        Optional summing matrix for reconciliation (if computed)
    tags
        Optional hierarchicalforecast package tags for series grouping
    """
    total_levels: int
    base_level: int
    top_level: int
    base_series_count: int
    top_series_count: int
    series_by_level: dict[int, list[dict[str, Any]]]
    summing_matrix: Optional[Any] = None
    tags: Optional[dict[str, Any]] = None


class FilteringSummary(BaseModel):
    """Metadata about the filtering process applied to time series.

    Parameters
    ----------
    excluded_levels_applied
        Set of hierarchy levels that were excluded in this filtering
    original_series_count
        Number of time series before filtering was applied
    filtered_series_count
        Total number of time series that were filtered out
    remaining_series_count
        Number of time series remaining after filtering
    filtered_ts_ids
        List of time series IDs that were filtered out
    filtered_by_level_count
        Number of series filtered due to level exclusions
    filtering_applied
        Whether any filtering was actually applied (True if any series were filtered)
    """
    excluded_levels_applied: set[str]
    original_series_count: int
    filtered_series_count: int
    remaining_series_count: int
    filtered_ts_ids: list[int]
    filtered_by_level_count: int
    filtering_applied: bool


class PeriodSummary(BaseModel):
    """Metadata about period clipping applied to actuals and forecasts.

    Parameters
    ----------
    final_actuals_length
        Number of datapoints in actuals after clipping and alignment
    final_forecasts_length
        Number of datapoints in forecasts after clipping and alignment
    actuals_period_config
        Configured period length for actuals (None = no limit)
    forecasts_period_config
        Configured period length for forecasts (None = no limit)
    """
    final_actuals_length: int
    final_forecasts_length: int
    actuals_period_config: Optional[int] = None
    forecasts_period_config: Optional[int] = None


class ReconciliationSummary(BaseModel):
    """Comprehensive metadata about the reconciliation process.

    Parameters
    ----------
    method_used
        The reconciliation method that was actually used
    methods_attempted
        List of all methods attempted (including failures)
    is_successful
        Whether the reconciliation completed successfully
    total_series_reconciled
        Number of time series that were reconciled
    processing_time_seconds
        Total time taken for reconciliation in seconds
    error_message
        Error message if reconciliation failed
    quality_metrics
        Optional dictionary of reconciliation quality metrics
    """
    method_used: ReconciliationMethod
    methods_attempted: list[ReconciliationMethod]
    is_successful: bool
    total_series_reconciled: int
    processing_time_seconds: float
    error_message: Optional[str] = None
    quality_metrics: Optional[dict[str, float]] = None


class ValidationResult(BaseModel):
    """Result of data validation for hierarchical forecasting.

    Parameters
    ----------
    is_valid
        Whether the data passes all validation checks
    errors
        List of validation errors that prevent processing
    warnings
        List of validation warnings that may affect quality
    """
    is_valid: bool
    errors: list[str]
    warnings: list[str]


class ErrorSummaryStatistics(BaseModel):
    """Statistical summary of percentage errors.

    Parameters
    ----------
    min
        Minimum percentage error value
    max
        Maximum percentage error value
    mean
        Mean percentage error value
    median
        Median percentage error value
    """
    min: float = Field(ge=0.0, description='Minimum percentage error')
    max: float = Field(ge=0.0, description='Maximum percentage error')
    mean: float = Field(ge=0.0, description='Mean percentage error')
    median: float = Field(ge=0.0, description='Median percentage error')


class SeriesInconsistency(BaseModel):
    """Individual time series inconsistency details for hierarchical consistency checking.

    Parameters
    ----------
    ts_id
        Time series identifier
    ts_name
        Human-readable time series name
    total_datapoints
        Total number of data points in the time series
    inconsistent_datapoints
        Number of data points with inconsistencies
    consistency_rate
        Percentage of consistent data points (0-100)
    series_mape
        Mean Absolute Percentage Error for this specific series
    summary_statistics
        Statistical summary of percentage errors (optional if no errors)
    contributing_base_series
        List of base series IDs that should sum to this aggregate series
    """
    ts_id: str
    ts_name: str
    total_datapoints: int
    inconsistent_datapoints: int
    consistency_rate: float
    series_mape: float
    summary_statistics: Optional[ErrorSummaryStatistics] = None
    contributing_base_series: list[str]


class ConsistencyCheckResult(BaseModel):
    """Result of hierarchical consistency check between actuals and summing matrix expectations.

    Parameters
    ----------
    total_aggregate_series
        Total number of aggregate series checked for consistency
    inconsistent_series_count
        Number of series with at least one inconsistent data point
    consistency_rate
        Percentage of series that are fully consistent (0-100)
    overall_mape
        Mean Absolute Percentage Error across all series and dates
    individual_series_inconsistencies
        List of per-series inconsistency details
    summary_statistics
        Statistical summary of percentage errors across all inconsistencies (optional if no errors)
    inconsistent_dates_ranked
        Dictionary mapping dates to count of inconsistent series, ordered by count descending
    """
    total_aggregate_series: int
    inconsistent_series_count: int
    consistency_rate: float
    overall_mape: float
    individual_series_inconsistencies: list[SeriesInconsistency]
    summary_statistics: Optional[ErrorSummaryStatistics] = None
    inconsistent_dates_ranked: dict[str, int]


class ConsistentForecastMetadata(BaseModel):
    """Consistent forecasts with nested original forecasts.

    Parameters
    ----------
    reconciliation_method
        Method used (bottom_up, top_down, etc.)
    model_selection_strategy
        Strategy for selecting models from forecasts
    hierarchy_structure
        Information about the hierarchical structure
    filtering_summary
        Summary of any filtering applied to the input data
    period_summary
        Summary of period clipping applied to actuals and forecasts
    reconciliation_summary
        Summary of the reconciliation process
    validation_summary
        Results of data validation checks
    consistency_check
        Results of hierarchical consistency analysis before reconciliation
    """
    reconciliation_method: ReconciliationMethod
    model_selection_strategy: Literal['best_ranking_model']
    hierarchy_structure: HierarchyMetadata
    filtering_summary: FilteringSummary
    period_summary: PeriodSummary
    reconciliation_summary: ReconciliationSummary
    validation_summary: ValidationResult
    consistency_check: ConsistencyCheckResult

    model_config = ConfigDict(
        protected_namespaces=()  # ignore warnings about field names starting with 'model_'
    )
