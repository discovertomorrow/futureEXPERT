import pytest

from futureexpert import ExpertClient
from futureexpert.pool import CheckInPoolResult, PoolCovDefinition


def test_get_pool_cov_overview___given_simple_parameters___returns_expected_data() -> None:
    # Arrange
    client = ExpertClient.from_dotenv()

    # Act
    indicator_overview = client.get_pool_cov_overview(granularity="Day", search="Business")

    # Assert
    assert any(indicator_overview.detailed_pool_cov_information.name.str.contains('Business'))


def test_get_pool_cov_overview___given_implausible_parameters___returns_error() -> None:
    # Arrange
    client = ExpertClient.from_dotenv()

    # Assert
    with pytest.raises(ValueError, match='No data found with the specified parameters'):
        client.get_pool_cov_overview(granularity="Day", search="implausiblekeyword")


def test_checkin_pool___get_valid_pool_cov_ids___creates_version() -> None:
    # Arrange
    client = ExpertClient.from_dotenv()
    indicator_overview = client.get_pool_cov_overview(granularity="Day")
    requested_pool_covs = []
    requested_pool_covs.append(PoolCovDefinition(
        pool_cov_id=indicator_overview.detailed_pool_cov_information.loc[0, "pool_cov_id"],
        version_id=indicator_overview.detailed_pool_cov_information.loc[0, "versions"][0]['version_id']))
    requested_pool_covs.append(PoolCovDefinition(
        pool_cov_id=indicator_overview.detailed_pool_cov_information.loc[0, "pool_cov_id"]))

    # Act
    checkin_result = client.check_in_pool_covs(requested_pool_covs=requested_pool_covs)

    # Assert
    assert checkin_result
    assert isinstance(checkin_result, CheckInPoolResult)


def test_query___get_valid_query___creates_expected_result() -> None:
    # Arrange
    client = ExpertClient.from_dotenv()
    indicator_overview = client.get_pool_cov_overview(granularity="Day")

    # Act
    queried_overview = indicator_overview.query(expr='name.str.contains("ferien")')

    # Assert
    assert len(queried_overview.detailed_pool_cov_information.index) >= 1
    assert queried_overview.detailed_pool_cov_information.name.str.contains('ferien').all()


def test_query___rerun_valid_query___creates_expected_result() -> None:
    # Arrange
    client = ExpertClient.from_dotenv()
    indicator_overview = client.get_pool_cov_overview(granularity="Day")

    # Act
    queried_overview = indicator_overview.query(expr='name.str.contains("ferien")')
    queried_overview = queried_overview.query(expr='name.str.contains("ferien")')

    # Assert
    assert len(queried_overview.detailed_pool_cov_information.index) >= 1
    assert queried_overview.detailed_pool_cov_information.name.str.contains('ferien').all()


def test_query___get_invalid_query___raises_error() -> None:
    # Arrange
    client = ExpertClient.from_dotenv()
    indicator_overview = client.get_pool_cov_overview(granularity="Day")

    # Assert
    with pytest.raises(ValueError, match='No data found after applying the filter.'):
        indicator_overview.query(expr='name=="notaname"')


def test_create_pool_cov_definitions___called____returns_expected_list() -> None:
    # Arrange
    client = ExpertClient.from_dotenv()
    indicator_overview = client.get_pool_cov_overview(granularity="Day")
    queried_overview = indicator_overview.query(expr='name.str.contains("ferien")')
    expected_ids = queried_overview.detailed_pool_cov_information.pool_cov_id.to_list()

    # Act
    definitions = queried_overview.create_pool_cov_definitions()

    # Assert
    assert len(definitions) == len(queried_overview.detailed_pool_cov_information.index)
    assert all(definition.pool_cov_id in expected_ids for definition in definitions)
