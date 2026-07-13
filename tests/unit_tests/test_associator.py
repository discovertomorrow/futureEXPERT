import pytest
from pydantic import ValidationError

from futureexpert.associator import AssociatorConfig


def test_AssociatorConfig___given_too_long_report_note___raises_error() -> None:
    with pytest.raises(ValidationError, match='String should have at most 255 characters'):
        AssociatorConfig(report_note='x' * 256)
