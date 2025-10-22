"""Contains the models with the configuration for the hierarchical reconciliation and the result format."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from futureexpert.forecast import ForecastValue, ModelStatus
from futureexpert.shared_models import BaseConfig

logger = logging.getLogger(__name__)


class ReconciliationMethod(str, Enum):
    """Reconciliation methods for hierarchical forecasting."""
    BOTTOM_UP = "bottom_up"
    TOP_DOWN_PROPORTION_AVERAGES = "top_down_proportion_averages"
    TOP_DOWN_FORECAST_PROPORTION = "top_down_forecast_proportion"
    MIN_TRACE_WLS_STRUCT = "min_trace_wls_struct"


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
    fallback_methods: List[ReconciliationMethod] = Field(default_factory=list)
    excluded_levels: List[str] = Field(default_factory=list)
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


class ForecastModel(BaseModel):
    """Single forecasting model results.

    Parameters
    ----------
    model_name
        Name of the forecasting model used (e.g., 'ARIMA', 'ETS', etc.)
    status
        Status of the forecast (Successful, Failed, etc.)
    forecasts
        List of forecast points with timestamps and values
    """
    model_name: Optional[str] = None
    status: ModelStatus
    forecasts: list[ForecastValue]


class OriginalForecastInput(BaseModel):
    """Input metadata for consistent forecast generation.

    Parameters
    ----------
    actuals
        Metadata about the actuals time series
    """
    actuals: ActualsMetadata


class OriginalForecast(BaseModel):
    """Original forecast result structure.

    This represents only the essential fields from the forecasting pipeline
    that are actually used during hierarchical reconciliation.

    Parameters
    ----------
    input
        Input metadata including actuals information
    models
        Non-empty list of model results (typically only first model is used)
    """
    input: OriginalForecastInput
    models: list[ForecastModel] = Field(min_length=1, description='Non-empty list of forecast models')


class ConsistentForecast(BaseModel):
    """Reconciled consistent forecast for a single time series with nested original forecast.

    Parameters
    ----------
    ts_id
        Time series identifier
    ts_name
        Human-readable time series name
    original_forecast
        The original forecast before reconciliation
    forecasts
        List of reconciled forecast points with timestamps and values
    """
    ts_id: int
    ts_name: str
    original_forecast: OriginalForecast
    forecasts: list[ForecastValue]


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


class ActualsMetadata(BaseModel):
    """Metadata about the actuals time series used for forecasting.

    Parameters
    ----------
    name
        Name of the time series
    grouping
        Hierarchical grouping information (empty dict for top level)
    """
    name: str
    grouping: dict[str, str] = Field(default_factory=dict)


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


class ConsistentForecastResult(BaseModel):
    """Consistent forecasts with nested original forecasts.

    Each reconciled forecast contains its corresponding original forecast as a nested field,
    eliminating the need for separate parallel arrays.

    Parameters
    ----------
    reconciled_forecasts
        List of hierarchically consistent, reconciled forecasts
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

    reconciled_forecasts: list[ConsistentForecast]
    reconciliation_method: ReconciliationMethod
    model_selection_strategy: Literal['best_ranking_model']
    hierarchy_structure: HierarchyMetadata
    filtering_summary: FilteringSummary
    period_summary: PeriodSummary
    reconciliation_summary: ReconciliationSummary
    validation_summary: ValidationResult
    consistency_check: ConsistencyCheckResult


def export_consistent_forecasts_to_pandas(results: ConsistentForecastResult) -> pd.DataFrame:
    """Export consistent forecasts.

    Parameters:
    -----------
    results
        Result after hierarchical reconciliation of forecasts.

    Returns:
    --------
    A DataFrame where each row represents the reconciled forecast information of a single timeseries of a certain date.
    """
    records = [
        {
            "name": reconciled_fc.ts_name,
            "time_stamp_utc": forecast_value.time_stamp_utc,
            "point_forecast_value": forecast_value.point_forecast_value,
        }
        for reconciled_fc in results.reconciled_forecasts
        for forecast_value in reconciled_fc.forecasts
    ]

    return pd.DataFrame(records)
