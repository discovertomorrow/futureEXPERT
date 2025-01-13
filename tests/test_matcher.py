import pytest
from pydantic import ValidationError

from futureexpert import ExpertClient, LagSelectionConfig, MatcherConfig


def test_create_matcher_payload___given_default_configuration___returns_start_and_end_dates_with_none() -> None:
    # Arrange
    config = MatcherConfig(title="Test", actuals_version="89sadhkasgdsadffff", covs_versions=["89sadhkasgdsadffff"])
    client = ExpertClient.from_dotenv()

    # Act
    payload = client._create_matcher_payload(config=config)

    # Assert
    assert payload['payload']['compute_config']['evaluation_start_date'] is None
    assert payload['payload']['compute_config']['evaluation_end_date'] is None


def test_create_matcher_payload___given_min_and_max_lag___returns_without_error() -> None:
    # Arrange
    config = MatcherConfig(title="Test",
                           actuals_version="89sadhkasgdsadffff",
                           covs_versions=["89sadhkasgdsadffff"],
                           lag_selection=LagSelectionConfig(min_lag=2, max_lag=8))
    client = ExpertClient()

    # Act
    client._create_matcher_payload(config=config)


@pytest.mark.parametrize("query", [
    ["BetterThanNoCov", "Lag <= 1", "Lag < 1"],
    ["Lag >= 1"],
    ["Lag > 1", "Lag == 1"],
    ["Lag != 1"],
    ["Lag == 11", "1 < Lag"],
    ["Lag==11", "Rank <= 1", "Rank < 1"],
    ["Rank >= 1"],
    ["Rank > 1", "Rank == 1"],
    ["Rank != 1", "Rank == 11", "1 < Rank"],
    ["Rank==11"]
])
def test_create_matcher_payload___given_valid_post_selection_parameters___returns_without_error(query: list[str]) -> None:

    # Arrange
    config = MatcherConfig(title="Test", actuals_version="89sadhkasgdsadffff", covs_versions=["89sadhkasgdsadffff"],
                           post_selection_queries=query)

    # Assert
    assert config.post_selection_queries == query


@pytest.mark.parametrize("invalid_query", [
    ["BetterThanNonCont"],
    ["sadfasdfBetterThanNoCovasdf"],
    ["Rank*..*8.2f1f3"],
    ["Rank*.2f1f3", "asasdfLagasd==dsdf8", "asasdfRankasd==dsdf8"]
])
def test_create_matcher_payload___given_incorrect_post_selection_parameters___raises_error(invalid_query: list[str]) -> None:

    # Arrange
    with pytest.raises(ValidationError) as exc_info:
        MatcherConfig(title="Test", actuals_version="89sadhkasgdsadffff", covs_versions=["]89sadhkasgdsadffff"],
                      post_selection_queries=invalid_query)

    # Assert
    assert 'The following post-selection queries are invalidly formatted' in str(exc_info.value)


@pytest.mark.parametrize("mixed_query", [
    (["BetterThanNonCont", "Lag > 1", "Lag == 1"]),
    (["Rank==11", "sadfasdfBetterThanNoCovasdf"])
])
def test_create_matcher_payload___given_mixed_post_selection_parameters___raises_error(mixed_query: list[str]) -> None:

    # Arrange
    with pytest.raises(ValidationError) as exc_info:
        MatcherConfig(title="Test", actuals_version="89sadhkasgdsadffff", covs_versions=["89sadhkasgdsadffff"],
                      post_selection_queries=mixed_query)

    # Assert
    assert 'The following post-selection queries are invalidly formatted' in str(exc_info.value)
