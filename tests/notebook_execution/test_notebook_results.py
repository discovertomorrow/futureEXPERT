import time

import pytest
from utility import extract_report_ids_from_notebook

from futureexpert.expert_client import ExpertClient


@pytest.mark.parametrize('path', [
    './notebooks/advanced_workflow.ipynb',
    './notebooks/associator.ipynb',
    './notebooks/checkin_configuration_options.ipynb',
    './notebooks/cov_matcher_and_forecast_monthly.ipynb',
    './notebooks/cov_matcher_and_forecast.ipynb',
    './notebooks/forecast_with_covariates.ipynb',
    './notebooks/make_forecasts_consistent.ipynb',
    './notebooks/getting_started.ipynb'
])
def test_notebook_results___confirm_all_run_ids_are_successful(path: str, expert_client: ExpertClient) -> None:
    report_ids = extract_report_ids_from_notebook(path)

    assert len(report_ids) > 0

    for report_id in report_ids:
        while not (status := expert_client.get_report_status(
            id=report_id,
            include_error_reason=False
        )).is_finished:
            # wait for unfinished report IDs
            time.sleep(10)

        assert status.progress.pending == 0
        assert status.results.error == 0
        assert status.results.no_evaluation == 0
