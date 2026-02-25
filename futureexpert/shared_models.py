"""Shared models used across multiple modules."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Generic, Literal, Optional, Sequence, TypeVar, Union

import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, WithJsonSchema


class PositiveInt(int):
    def __new__(cls, value: int) -> PositiveInt:
        if value < 1:
            raise ValueError('The value must be a positive integer.')
        return super().__new__(cls, value)


ValidatedPositiveInt = Annotated[PositiveInt,
                                 BeforeValidator(lambda x: PositiveInt(int(x))),
                                 # raises an error without the lambda wrapper
                                 PlainSerializer(lambda x: int(x), return_type=int),
                                 WithJsonSchema({'type': 'integer', 'minimum': 1})]

MAX_TS_LEN_CONFIG = {
    'halfhourly': 302,
    'hourly': 252,
    'daily': 546,
    'weekly': 326,
    'monthly': 86,
    'quarterly': 38,
    'yearly': 26,
}


class BaseConfig(BaseModel):
    """Base configuration that is used for most models."""
    model_config = ConfigDict(allow_inf_nan=False,
                              extra='forbid',
                              arbitrary_types_allowed=True)


class TimeSeriesValue(BaseModel):
    """Value of a time series.

    Parameters
    ----------
    time_stamp_utc
        The time stamp of the value.
    value
        The value.
    """
    time_stamp_utc: datetime
    value: Annotated[float, Field(allow_inf_nan=False)]


class TimeSeries(BaseModel):
    """Time series data.

    Parameters
    ----------
    name
        Name of the time series.
    group
        Group of the time series.
    granularity
        Granularity of the time series.
    values
        The actual values of the time series.
    unit
        The unit of the time .
    unit_factors
        Factors to convert the time series in another unit.
    grouping
        Hierarchy levels ot the time series.
    """
    name: Annotated[str, Field(min_length=1)]
    group: str
    granularity: Annotated[str, Field(min_length=1)]
    values: Annotated[Sequence[TimeSeriesValue], Field(min_length=1)]
    unit: Optional[str] = None
    unit_factors: Optional[dict[str, float]] = None
    grouping: Optional[dict[str, Union[str, int]]] = None


class CovariateRef(BaseModel):
    """Covariate reference.

    Parameters
    ----------
    name
        Name of the Covariate
    lag
        Lag by which the covariate was used.
    """
    name: str
    lag: int


class Covariate(BaseModel):
    """Covariate.

    Parameters
    ----------
    ts
        Time series object of the covariate. Not lagged.
    lag
        Lag by which the covariate was used.
    """
    ts: TimeSeries
    lag: int


RerunStatus = Literal['Error', 'NoEvaluation']


class ReportStatusProgress(BaseModel):
    """Progress of a forecasting report."""
    requested: int
    pending: int
    finished: int


class ReportStatusResults(BaseModel):
    """Result status of a forecasting report.

    This only includes runs that are already finished."""
    successful: int
    no_evaluation: int
    error: int


class ErrorReason(BaseModel):
    """Details about a specific error in a report.

    Parameters
    ----------
    status
        The status of the run ('Error' or 'NoEvaluation').
    error_message
        The error message describing what went wrong.
    timeseries
        List of time series names that encountered this error.
    """
    status: str
    error_message: Optional[str]
    timeseries: list[str]

    @staticmethod
    def parse_error_reasons(customer_specific: dict[str, Any]) -> list[ErrorReason]:
        """Creates error reasons from raw customer_specific object."""
        log_messages = customer_specific.get('log_messages', None)
        assert log_messages is not None, 'missing log_messages property in customer_specific'
        assert isinstance(log_messages, list), 'unexpected type of log_messages'
        return [ErrorReason.model_validate(msg) for msg in log_messages]


class ReportStatus(BaseModel):
    """Status of a forecast or matcher report.

    Parameters
    ----------
    id
        The identifier of the report.
    description
        The description of the report.
    result_type
        The result type of the report.
    progress
        Progress summary of the report.
    results
        Success/error summary of the report.
    error_reasons
        Details about the errors of the report. Each error reason contains the status,
        error message, and list of affected time series.
    prerequisites
        If the status was requested for a report that depends on other reports (ChainedReportIdentifier)
        all other report statuses are contained in the prerequisites in order to get an easy overview.
    """
    id: ReportIdentifier
    description: str
    result_type: str
    progress: ReportStatusProgress
    results: ReportStatusResults
    error_reasons: Optional[list[ErrorReason]] = None
    prerequisites: list[ReportStatus] = Field(default_factory=list)

    @property
    def is_finished(self) -> bool:
        """Indicates whether a forecasting report is finished."""
        return self.progress.pending == 0

    def print(self, print_prerequisites: bool = True, print_error_reasons: bool = True) -> None:
        """Prints a summary of the status.

        Parameters
        ----------
        print_prerequisites
            Enable or disable printing of prerequisite reports.
        print_error_reasons
            Enable or disable printing of error reasons.
        """
        title = f'Status of report "{self.description}" of type "{self.result_type}":'
        run_description = 'time series' if self.result_type in ['forecast', 'matcher'] else 'runs'
        if print_prerequisites:
            for prerequisite in self.prerequisites:
                prerequisite.print(print_error_reasons=print_error_reasons)

        if self.progress.requested == 0:
            print(f'{title}\n  No {run_description} created')
            return

        pct_txt = f'{round(self.progress.finished/self.progress.requested*100)} % are finished'
        overall = f'{self.progress.requested} {run_description} requested for calculation'
        finished_txt = f'{self.progress.finished} {run_description} finished'
        noeval_txt = f'{self.results.no_evaluation} {run_description} without evaluation'
        error_txt = f'{self.results.error} {run_description} ran into an error'
        print(f'{title}\n {pct_txt} \n {overall} \n {finished_txt} \n {noeval_txt} \n {error_txt}')

        if print_error_reasons and self.error_reasons is not None and len(self.error_reasons) > 0:
            print('\nError reasons:')
            for error_reason in self.error_reasons:
                ts_count = len(error_reason.timeseries)
                ts_names = ', '.join(error_reason.timeseries[:3])  # Show first 3 time series
                if ts_count > 3:
                    ts_names += f' ... and {ts_count - 3} more'
                print(f'  [{error_reason.status}] {error_reason.error_message if error_reason.error_message else ""}')
                print(f'    Affected time series ({ts_count}): {ts_names}')


class ReportIdentifier(BaseModel):
    """Report ID and Settings ID of a report. Required to identify the report, e.g. when retrieving the results."""
    report_id: int
    settings_id: Optional[int]


class ChainedReportIdentifier(ReportIdentifier):
    """Extended report identifier with prerequisites."""
    prerequisites: list[ReportIdentifier]

    @classmethod
    def of(cls, final_report_identifier: ReportIdentifier, prerequisites: list[ReportIdentifier]) -> ChainedReportIdentifier:
        return cls(report_id=final_report_identifier.report_id,
                   settings_id=final_report_identifier.settings_id,
                   prerequisites=prerequisites)


class ReportSummary(BaseModel):
    """Report ID and description of a report."""
    report_id: int
    description: str
    result_type: str


TBaseModel = TypeVar('TBaseModel', bound=BaseModel)


class PydanticModelList(Generic[TBaseModel], list[TBaseModel]):
    def to_df(self) -> pd.DataFrame:
        """Converts the list into a Pandas Data Frame."""
        return pd.DataFrame([item.model_dump() for item in self])
