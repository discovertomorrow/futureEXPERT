from typing import Optional, Sequence, Union

from pydantic import BaseModel

from futureexpert.forecast import ForecastValue
from futureexpert.shared_models import BaseConfig, Covariate, CovariateRef, TimeSeries, TimeSeriesValue


class Scenario(BaseConfig):
    """Configuration of scenarios for one covariate.

    Parameters
    ----------
    ts
        The original covariate time series with lag information.
    ts_version
        Version of the covariate.
    high
        High scenario values.
    low
        Low scenario values.
    custom
        Custom scenario values (Optional).
    """
    ts: Union[Covariate, CovariateRef]
    ts_version: str
    high: list[TimeSeriesValue]
    low: list[TimeSeriesValue]
    custom: Optional[list[TimeSeriesValue]] = None

    def add_custom_values(self, values: list[float]) -> None:
        """Add custom scenario values to the Scenario."""
        if len(values) != len(self.high):
            raise ValueError('All Scenarios need the same length')
        self.custom = [TimeSeriesValue(time_stamp_utc=x.time_stamp_utc, value=values[idx])
                       for idx, x in enumerate(self.high)]


class ScenarioValuesConfig(BaseConfig):
    """Configuration for scenario values creation.

    Parameters
    ----------
    actuals_version
        Version from the time series that should get forecasted.
    actuals_name
        Name of the actuals time series.
    covariate_versions
        All the covariate versions from the covariates listed in `covariates`.
    covariates
        List of covariates to create scenarios for.
    fc_horizon
        The number of forecast periods.
    """
    actuals_version: str
    actuals_name: str
    covariate_versions: list[str]
    covariates: list[CovariateRef]
    fc_horizon: int


class ShaperConfig(BaseConfig):
    """Configuration of a shaper run.

    Parameters
    ----------
    report_note
        Title of the report.
    actuals_version
        Version of the actuals.
    actuals_name
        Name of the time series.
    scenarios
        Information about the covariates and their scenario values.
    db_name
        Only accessible for internal use. Name of the database to use for storing the results.
    """
    report_note: str
    actuals_version: str
    actuals_name: str
    scenarios: list[Scenario]
    db_name: Optional[str] = None


class ResultScenario(BaseConfig):
    """Configuration of scenarios for one covariate.

    Parameters
    ----------
    ts
        Information about the covariate.
    high
        High scenario values.
    low
        Low scenario values.
    custom
        Custom scenario values (Optional).
    """

    ts: Covariate
    high: Sequence[TimeSeriesValue]
    low: Sequence[TimeSeriesValue]
    custom: Optional[Sequence[TimeSeriesValue]] = None


class ShaperInput(BaseModel):
    """Input of the shaper service.

    Parameters
    ----------
    actuals
        Time series for which the forecasts where performed.
    scenarios
        The Scenario Information.
    """
    actuals: TimeSeries
    scenarios: list[ResultScenario]


class ShaperResult(BaseModel):
    """Result of the shaper service.

   Parameters
    ----------
    input
        Input Information.
    forecast_low
        Forecast values of the low scenario.
    forecast_high
        Forecast values of the high scenario.
    forecast_custom
        Forecast values of the custom scenario.
    """
    input: ShaperInput
    forecast_low: Sequence[ForecastValue]
    forecast_high: Sequence[ForecastValue]
    forecast_custom: Sequence[ForecastValue]
