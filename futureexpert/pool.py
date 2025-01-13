from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

import pandas as pd
from pydantic import BaseModel, model_validator
from typing_extensions import Self


class PoolCovOverview():
    """Contains all functionality to inspect, view and process the POOL covariates."""

    def __init__(self, overview_json: dict[Any, Any]) -> None:
        """Initializer.

        Parameters
        ----------
        overview_json
            Covariate raw data.
        """
        self.detailed_pool_cov_information = pd.DataFrame(overview_json).rename(
            columns={"distance": "granularity", 'indicator_id': 'pool_cov_id'})

    @property
    def pool_cov_information(self) -> pd.DataFrame:
        """Allows to access the covariates reduced to the most important columns (name, description, pool_cov_id)"""
        return self.detailed_pool_cov_information.loc[:, ['name', 'description', 'pool_cov_id']]

    def get_versions_of_pool_cov(self, pool_cov: pd.Series) -> pd.DataFrame:
        """Returns the version information of a single pool_cov.

        Parameters
        ----------
        pool_cov
            One row representing a single pool_cov of the detailed_pool_cov_information.

        Returns
        ------
        Table with the following columns
            name, description, pool_cov_id: meta information about the selected pool_cov.
            reference_time: The timestamp at which the data was available. Usually very similar to created_at.
            created_at: The timestamp at which the pool_cov data was accessed.
            first_observation: The timestamp of the earliest observation.
            final_observation: The timestamp of the latest oversevation.
        """
        versions = pd.DataFrame(pool_cov["versions"])

        # Create new columns with 'name' and 'description'
        new_columns = pd.DataFrame({
            'name': pool_cov['name'],
            'description': pool_cov['description'],
            'pool_cov_id': pool_cov['pool_cov_id']
        }, index=versions.index)

        # Concatenate new columns with existing DataFrame
        return pd.concat([new_columns, versions], axis=1)

    def query(self, expr: str) -> PoolCovOverview:
        """Use a query expression to filter the data.

        Parameters
        ----------
        expr
            Query expression that is used to filter the data. See pandas.DataFrame.query() for more information.

        Returns
        -------
        A new instance of PoolCovOverview with only the filtered data.
        """
        queried_df = self.detailed_pool_cov_information.query(expr).reset_index(drop=True)
        if len(queried_df.index) == 0:
            raise ValueError('No data found after applying the filter.')
        return PoolCovOverview(queried_df.to_records(index=False))

    def create_pool_cov_definitions(self) -> list[PoolCovDefinition]:
        """Creates a list of definitions for all covariates in the data. This list
        can then be used to create a version via check_in_pool,

        Returns
        -------
        List of definitions.
        """
        pool_cov_information = (self.detailed_pool_cov_information
                                .loc[:, ['pool_cov_id']]
                                .to_dict('records'))
        return [PoolCovDefinition.model_validate(pool_cov) for pool_cov in pool_cov_information]


class PoolCovDefinition(BaseModel):
    """Definition of a single requested pool_cov.

    Parameters
    ----------
    pool_cov_id
        ID of a the pool_cov as it is found on POOL.
    version_id
        ID of one specific version. If not defined, the newest version
        of the pool_cov will be returned.
    """
    pool_cov_id: str
    version_id: Optional[str] = None

    @model_validator(mode='after')
    def _validate_ids(self) -> Self:
        UUID(self.pool_cov_id)
        if self.version_id:
            UUID(self.version_id)
        return self


class CheckInPoolResult(BaseModel):
    """Result of the check in pool process.

    Parameters
    ----------
    time_series_metadata
        Metadata for all time series that were uploaded.
    version_id
        Version under which the time series have been stored.
    """
    time_series_metadata: list[dict[Any, Any]]
    version_id: str
