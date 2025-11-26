"""Contains the models with the configuration for the associator and the result format."""
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from pydantic import Field, model_validator

from futureexpert.shared_models import BaseConfig, TimeSeries


class DataSelection(BaseConfig):
    """Time series selection.

    Parameters
    ----------
    version
        Time series version to be used. If None, then the latest version is used.
    filter
        Filter to select a subset of time series based on their metadata.
    """

    version: Optional[str] = None
    filter: dict[str, Any] = Field(default_factory=dict)


class TrendDetectionConfiguration(BaseConfig):
    """Configuration for trend detection.

    Parameters
    ----------
    end_time
        End (inclusive) of the time span used for trend detection.
    max_number_of_obs
        Width of the time span used for trend detection; (leading and trailing) missing values
        are disregarded, that is, at most this number of observations are used for a given time series.
    number_of_nans_tolerated
        Leading and lagging missing values are dropped prior to running the trend detection; if this
        results in a loss of more than this number of observations lost, then the trend is considered
        undetermined.
    """
    end_time: Optional[datetime] = None
    max_number_of_obs: int = Field(default=6, gt=0)
    number_of_nans_tolerated: int = 2


class ClusteringConfiguration(BaseConfig):
    """Configuration for clustering.

    If start_time or end_time is not provided, then the missing(s) of the two will be
    determined automatically; the final four parameters govern this process.

    Parameters
    ----------
    create_clusters
        If True, then the service will attempt clustering.
    n_clusters
        Number of clusters of complete and non-constant time series.
    start_time
        Observations from start_time (inclusive) onwards will be considered during clustering.
    end_time
        Observations up to end_time (inclusive) will be considered during clustering.
    """
    create_clusters: bool = True
    n_clusters: Optional[int] = Field(default=None, gt=0)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @model_validator(mode='after')
    def validate_times(self) -> 'ClusteringConfiguration':
        if self.start_time is not None and self.end_time is not None and self.start_time > self.end_time:
            raise ValueError('End time precedes start time.')
        return self


class AssociatorConfig(BaseConfig):
    """Service configuration.

    Parameters
    ----------
    data_selection
        Configuration on the selection of time series used for carrying out the service.
    trend_detection
        Configuration for trend detection.
    clustering
        Configuration for clustering.
    report_note
        User-defined string to be included in the report.
    db_name
        Only accessible for internal use. Name of the database to use for storing the results.
    """

    data_selection: DataSelection = Field(default_factory=DataSelection)
    trend_detection: TrendDetectionConfiguration = Field(default_factory=TrendDetectionConfiguration)
    clustering: ClusteringConfiguration = Field(default_factory=ClusteringConfiguration)
    report_note: str
    db_name: Optional[str] = None


class TrendLabel(BaseConfig):
    """Trend label for a time series.

    Parameters
    ----------
    ts_name
        Name of the time series.
    trend_label
        Trend label.
    slope
        Slope of the trend.
    num_obs
        Number of observations used for trend detection.
    """
    ts_name: str
    trend_label: str
    slope: Optional[float]
    num_obs: Optional[int]


class TrendResult(BaseConfig):
    """Result of trend detection.

    Parameters
    ----------
    start_time
        Start of the time span used for trend detection.
    end_time
        End of the time span used for trend detection.
    trend_labels
        List of trend labels for the time series.
    """
    start_time: datetime
    end_time: datetime
    trend_labels: list[TrendLabel]


class Group(BaseConfig):
    """A group definition to which time series can be assigned.

    Parameters
    ----------
    group_id
        ID of the group.
    group_name
        Name of the group.
    group_size
        Number of time series in the group.
    """
    group_id: int
    group_name: str
    group_size: int


class GroupAssignment(BaseConfig):
    """Assignment of a time series to a group.

    Parameters
    ----------
    ts_name
        Name of the time series.
    group_id
        ID of the group the time series is assigned to.
    """
    ts_name: str
    group_id: int


class ClusteringResult(BaseConfig):
    """Result of clustering.

    Parameters
    ----------
    start_time
        Start of the time span used for clustering.
    end_time
        End of the time span used for clustering.
    groups
        List of groups.
    group_assignments
        List of group assignments.
    """
    start_time: datetime
    end_time: datetime
    groups: list[Group]
    group_assignments: list[GroupAssignment]


class AssociatorResult(BaseConfig):
    """Result of the associator service.

    Parameters
    ----------
    input
        List of time series used as input.
    trend
        Result of trend detection.
    clustering
        Result of clustering.
    """
    input: list[TimeSeries]
    trend: TrendResult
    clustering: ClusteringResult


def export_associator_results_to_pandas(results: AssociatorResult) -> pd.DataFrame:
    """Export associator results to a pandas DataFrame.

    Parameters
    ----------
    results
        Associator results.

    Returns
    -------
    A pandas DataFrame with columns 'ts_name', 'group_id', 'trend_label', 'slope', 'num_obs'.
    """

    cluster_df = pd.DataFrame.from_records([x.model_dump() for x in results.clustering.group_assignments])
    trend_df = pd.DataFrame.from_records([x.model_dump() for x in results.trend.trend_labels])
    return cluster_df.merge(trend_df, how='outer', on='ts_name', validate='1:1')
