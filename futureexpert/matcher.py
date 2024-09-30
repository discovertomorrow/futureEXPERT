"""Contains the models with the configuration for the matcher and the result format."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from futureexpert.pool import PoolCovDefinition
from futureexpert.shared_models import BaseConfig, Covariate, CovariateRef, TimeSeries, ValidatedPositiveInt


class MatcherConfig(BaseConfig):
    """Configuration for a futureMATCHER run.

    Parameters
    ----------
    title
        A short description of the report.
    actuals_version
        The version ID of the actuals.
    covs_version
        The version of the covariates.
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
    lag_selection_fixed_lags
        Lags that are tested in the lag selection.
    lag_selection_min_lag
        Minimal lag that is tested in the lag selection. For example, a lag 3 means the covariate
        is shifted 3 data points into the future.
    lag_selection_max_lag
        Maximal lag that is tested in the lag selection. For example, a lag 12 means the covariate
        is shifted 12 data points into the future.
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
    lag_selection_fixed_season_length
        An optional parameter specifying the length of a season in the dataset.
    pool_covs
        List of covariate definitions.
    db_name
        Only accessible for internal use. Name of the database to use for storing the results.
    """
    title: str
    actuals_version: str
    covs_version: Optional[str] = None
    actuals_filter: dict[str, Any] = Field(default_factory=dict)
    covs_filter: dict[str, Any] = Field(default_factory=dict)
    lag_selection_fixed_lags: Optional[list[int]] = None
    lag_selection_min_lag: Optional[int] = None
    lag_selection_max_lag: Optional[int] = None
    evaluation_start_date: Optional[str] = None
    evaluation_end_date: Optional[str] = None
    max_publication_lag: int = 2
    post_selection_queries: list[str] = []
    enable_leading_covariate_selection: bool = True
    lag_selection_fixed_season_length: Optional[int] = None
    pool_covs: Optional[list[PoolCovDefinition]] = None
    db_name: Optional[str] = None

    @model_validator(mode='after')
    def _check_lag_selection_range(self) -> Self:

        min_lag = self.lag_selection_min_lag
        max_lag = self.lag_selection_max_lag

        if (min_lag is None) ^ (max_lag is None):
            raise ValueError(
                'If one of `lag_selection_min_lag` and `lag_selection_max_lag` is set the other one also needs to be set.')
        if min_lag and max_lag:
            if self.lag_selection_fixed_lags is not None:
                raise ValueError('Fixed lags and min/max lag are mutually exclusive.')
            if max_lag < min_lag:
                raise ValueError('lag_selection_max_lag needs to be higher as lag_selection_min_lag.')
            lag_range = abs(max_lag - min_lag)
            if lag_range > 15:
                raise ValueError(f'Only a range of 15 lags is allowed to test. The current range is {lag_range}.')

        return self

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

    @model_validator(mode="after")
    def _exactly_one_covariate_definition(self) -> Self:
        fields = [
            'covs_version',
            'pool_covs'
        ]

        set_fields = [field for field in fields if getattr(self, field) is not None]

        if len(set_fields) != 1:
            raise ValueError(f"Exactly one of {', '.join(fields)} can be set. Found: {', '.join(set_fields)}")

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
