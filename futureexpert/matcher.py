"""Contains the models with the configuration for the matcher and the result format."""
from __future__ import annotations

from typing import Annotated, Any, Optional

import pandas as pd
import pydantic
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from futureexpert.pool import PoolCovDefinition
from futureexpert.shared_models import (BaseConfig,
                                        Covariate,
                                        CovariateRef,
                                        RerunStatus,
                                        TimeSeries,
                                        ValidatedPositiveInt)


class LagSelectionConfig(BaseModel):
    """Configures covariate lag selection.

    Parameters
    ----------
    fixed_lags
        Lags that are tested in the lag selection.
    min_lag
        Minimal lag that is tested in the lag selection. For example, a lag 3 means the covariate
        is shifted 3 data points into the future.
    max_lag
        Maximal lag that is tested in the lag selection. For example, a lag 12 means the covariate
        is shifted 12 data points into the future.
    """
    min_lag: Optional[int] = None
    max_lag: Optional[int] = None
    fixed_lags: Optional[list[int]] = None

    @model_validator(mode='after')
    def _check_range(self) -> Self:
        if (self.min_lag is None) ^ (self.max_lag is None):
            raise ValueError(
                'If one of `min_lag` and `max_lag` is set the other one also needs to be set.')

        if self.min_lag and self.max_lag:
            if self.fixed_lags is not None:
                raise ValueError('Fixed lags and min/max lag are mutually exclusive.')
            if self.max_lag < self.min_lag:
                raise ValueError('max_lag needs to be greater or equal to min_lag.')
            lag_range = abs(self.max_lag - self.min_lag) + 1
            if lag_range > 15:
                raise ValueError(f'Only 15 lags are allowed to be tested. The requested range has length {lag_range}.')

        if self.fixed_lags and len(self.fixed_lags) > 15:
            raise ValueError(
                f'Only 15 lags are allowed to be tested. The provided fixed lags has length {len(self.fixed_lags)}.')

        return self


class MatcherConfig(BaseConfig):
    """Configuration for a MATCHER run.

    Parameters
    ----------
    title
        A short description of the report.
    actuals_version
        The version ID of the actuals.
    covs_versions
        List of versions of the covariates.
    actuals_filter
        Filter criterion for actuals time series. The given actuals version is
        automatically added as additional filter criterion. Possible Filter criteria are all fields that are part
        of the TimeSeries class. e.g. {'name': 'Sales'}
        For more complex filter check: https://www.mongodb.com/docs/manual/reference/operator/query/#query-selectors
    covs_filter
        Filter criterion for covariates time series. The given covariate version is
        automatically added as additional filter criterion. Possible Filter criteria are all fields that are part
        of the TimeSeries class. e.g. {'name': 'Sales'}
        For more complex filter check: https://www.mongodb.com/docs/manual/reference/operator/query/#query-selectors
    max_ts_len
        At most this number of most recent observations of the actuals time series is used. Check the variable MAX_TS_LEN_CONFIG
        for allowed configuration.
    lag_selection
        Configuration of covariate lag selection.
    evaluation_start_date
        Optional start date for the evaluation. The input should be in the ISO format
        with date and time, "YYYY-mm-DDTHH-MM-SS", e.g., "2024-01-01T16:40:00".
        Actuals and covariate observations prior to this start date are dropped.
    evaluation_end_date
        Optional end date for the evaluation. The input should be in the ISO format
        with date and time, "YYYY-mm-DDTHH-MM-SS", e.g., "2024-01-01T16:40:00".
        Actuals and covariate observations after this end date are dropped.
    max_publication_lag
        Maximal publication lag for the covariates. The publication lag of a covariate
        is the number of most recent observations (compared to the actuals) that are
        missing for the covariate. E.g., if the actuals (for monthly granularity) end
        in April 2023 but the covariate ends in February 2023, the covariate has a
        publication lag of 2.
    associator_report_id
        Optional report id of clustering results. If None, the database is searched for a fitting clustering.
        The clustering results are used in the post-selection. If there are too many selected behind this is
        that they all would give similar results in forecasting. Only used if `use_clustering_results`
        is true.
    use_clustering_results
        If true clustering results are used.
    post_selection_queries
        List of queries that are executed on the ranking summary DataFrame. Only ranking entries that
        match the queries are kept. The query strings need to satisfy the pandas query syntax
        (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html). Here are the columns
        of the ranking summary DataFrame that you might want to filter on:

        Column Name          |      Data Type   |    Description
        -----------------------------------------------------------------------------------------------
        Lag                  |          Int64   |    Lag of the covariate.
        Rank                 |        float64   |    Rank of the model.
        BetterThanNoCov      |           bool   |    Indicates whether the model is better than the non-cov model.
    enable_leading_covariate_selection
        When True, all covariates after the lag is applied that do not have at least one more
        datapoint beyond the the time period covered by actuals are removed from the candidate
        covariates passed to covariate selection.
    fixed_season_length
        An optional parameter specifying the length of a season in the dataset.
    pool_covs
        List of covariate definitions.
    db_name
        Only accessible for internal use. Name of the database to use for storing the results.
    rerun_report_id
        ReportId from which failed runs should be recomputed.
        Ensure to use the same ts_version. Otherwise all time series get computed again.
    rerun_status
        Status of the runs that should be computed again. `Error` and/or `NoEvaluation`.
    """
    title: str
    actuals_version: str
    covs_versions: list[str] = Field(default_factory=list)
    actuals_filter: dict[str, Any] = Field(default_factory=dict)
    covs_filter: dict[str, Any] = Field(default_factory=dict)
    max_ts_len: Annotated[
        Optional[int], pydantic.Field(ge=1, le=1500)] = None
    lag_selection: LagSelectionConfig = LagSelectionConfig()
    evaluation_start_date: Optional[str] = None
    evaluation_end_date: Optional[str] = None
    max_publication_lag: int = 2
    associator_report_id: Optional[pydantic.PositiveInt] = None
    use_clustering_results: bool = False
    post_selection_queries: list[str] = []
    enable_leading_covariate_selection: bool = True
    fixed_season_length: Optional[int] = None
    pool_covs: Optional[list[PoolCovDefinition]] = None
    db_name: Optional[str] = None
    rerun_report_id: Optional[int] = None
    rerun_status: list[RerunStatus] = ['Error']

    @model_validator(mode='after')
    def _validate_post_selection_queries(self) -> Self:
        # Validate the post-selection queries.
        invalid_queries = []
        columns = {
            'Lag': 'int',
            'Rank': 'float',
            'BetterThanNoCov': 'bool'
        }
        # Create an empty DataFrame with the specified column names and data types
        validation_df = pd.DataFrame(columns=columns.keys()).astype(columns)
        for postselection_query in self.post_selection_queries:
            try:
                validation_df.query(postselection_query, )
            except Exception:
                invalid_queries.append(postselection_query)

        if len(invalid_queries):
            raise ValueError("The following post-selection queries are invalidly formatted: "
                             f"{', '.join(invalid_queries)}. ")

        return self

    @model_validator(mode='after')
    def _validate_rerun_report_id(self) -> Self:

        if self.rerun_report_id is not None and self.pool_covs is not None:
            raise ValueError('rerun_report_id can not be used with pool_covs. '
                             'Use the exact covs_version used in the rerun_report_id.')

        return self


class CovariateRankingDetails(BaseModel):
    """Final rank for a given set of covariates.

    Parameters
    ----------
    rank
        Rank for the given set of covariates.
    covariates
        Used covariates (might be zero or more than one).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rank: ValidatedPositiveInt
    covariates: list[Covariate]


class ActualsCovsConfiguration(BaseModel):
    """Configuration of actuals and covariates via name and lag.

    Parameters
    ----------
    actuals_name
        Name of the time series.
    covs_configurations
        List of Covariates.
    """
    actuals_name: str
    covs_configurations: list[CovariateRef]


class MatcherResult(BaseModel):
    """Result of a covariate matcher run and the corresponding input data.

    Parameters
    ----------
    actuals
        Time series for which the matching was performed.
    ranking
        Ranking of the different covariate and non-covariate models.
    """
    actuals: TimeSeries
    ranking: list[CovariateRankingDetails]

    def convert_ranking_to_forecast_config(self) -> ActualsCovsConfiguration:
        """Converts MATCHER results into the input format for the FORECAST."""
        covs_config = [CovariateRef(name=cov.ts.name, lag=cov.lag) for r in self.ranking for cov in r.covariates]
        return ActualsCovsConfiguration(actuals_name=self.actuals.name,
                                        covs_configurations=covs_config)
