import json
import re

import pytest

from futureexpert.expert_client import ExpertClient


@pytest.mark.parametrize("path", [
    "./notebooks/advanced_workflow.ipynb",
    "./notebooks/checkin_configuration_options.ipynb",
    "./notebooks/cov_matcher_and_forecast_monthly.ipynb",
    "./notebooks/cov_matcher_and_forecast.ipynb",
    "./notebooks/forecast_with_covariates.ipynb",
    "./notebooks/make_forecasts_consistent.ipynb",
    "./notebooks/getting_started.ipynb"
])
def test_notebook_results___confirm_all_run_ids_are_successful(path: str, expert_client: ExpertClient) -> None:

    with open(path, 'r') as o:
        data = json.load(o)
    html_string = ''
    for cell in data.get("cells", []):
        if cell['cell_type'] == 'markdown':
            continue
        output = [''.join(out.get('text', '')) for out in cell['outputs']]
        html_string = html_string + ''.join(output)

    report_ids = re.findall(r"Report created with ID (\d+)", html_string)

    assert len(report_ids) > 0

    for report_id in report_ids:
        status = expert_client.get_report_status(id=report_id, include_error_reason=False)

        assert status.progress.pending == 0
        assert status.results.error == 0
        assert status.results.no_evaluation == 0
