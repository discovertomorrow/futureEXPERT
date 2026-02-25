from futureexpert.shared_models import (ErrorReason,
                                        ReportIdentifier,
                                        ReportStatus,
                                        ReportStatusProgress,
                                        ReportStatusResults)


class TestReportStatus:
    """Tests for ReportStatus.print() method with various error scenarios."""

    def _create_report_status(self, error_reasons=None, num_errors=0):
        """Helper to create a ReportStatus instance."""
        return ReportStatus(
            id=ReportIdentifier(report_id=123, settings_id=456),
            description='Test Report',
            result_type='forecast',
            progress=ReportStatusProgress(requested=10, pending=0, finished=10),
            results=ReportStatusResults(successful=10 - num_errors, no_evaluation=0, error=num_errors),
            error_reasons=error_reasons
        )

    def test_print___with_single_error_reason___prints_error(self, capsys):
        """Test printing with a single error affecting multiple time series."""
        # Arrange
        error_reasons = [
            ErrorReason(
                status='Error',
                error_message='Insufficient data points for forecasting',
                timeseries=['sales_01', 'sales_02', 'sales_03']
            )
        ]
        status = self._create_report_status(error_reasons=error_reasons, num_errors=3)

        # Act
        status.print()

        # Assert
        captured = capsys.readouterr()
        assert 'Error reasons:' in captured.out
        assert '[Error] Insufficient data points for forecasting' in captured.out
        assert 'Affected time series (3): sales_01, sales_02, sales_03' in captured.out
        assert '3 time series ran into an error' in captured.out

    def test_print___with_multiple_error_reasons___prints_all_errors(self, capsys):
        """Test printing with multiple different errors."""
        # Arrange
        error_reasons = [
            ErrorReason(
                status='Error',
                error_message='Insufficient data points',
                timeseries=['sales_01', 'sales_02']
            ),
            ErrorReason(
                status='NoEvaluation',
                error_message='Missing covariate data',
                timeseries=['sales_03']
            ),
            ErrorReason(
                status='Error',
                error_message='Invalid date range',
                timeseries=['sales_04', 'sales_05']
            )
        ]
        status = self._create_report_status(error_reasons=error_reasons, num_errors=5)

        # Act
        status.print()

        # Assert
        captured = capsys.readouterr()
        assert 'Error reasons:' in captured.out
        assert '[Error] Insufficient data points' in captured.out
        assert 'Affected time series (2): sales_01, sales_02' in captured.out
        assert '[NoEvaluation] Missing covariate data' in captured.out
        assert 'Affected time series (1): sales_03' in captured.out
        assert '[Error] Invalid date range' in captured.out
        assert 'Affected time series (2): sales_04, sales_05' in captured.out

    def test_print___with_many_affected_timeseries___truncates_display(self, capsys):
        """Test that long lists of time series are truncated properly."""
        # Arrange
        error_reasons = [
            ErrorReason(
                status='Error',
                error_message='Database connection timeout',
                timeseries=[f'ts_{i:03d}' for i in range(10)]  # 10 time series
            )
        ]
        status = self._create_report_status(error_reasons=error_reasons, num_errors=10)

        # Act
        status.print()

        # Assert
        captured = capsys.readouterr()
        assert 'Error reasons:' in captured.out
        assert '[Error] Database connection timeout' in captured.out
        assert 'Affected time series (10): ts_000, ts_001, ts_002 ... and 7 more' in captured.out

    def test_print___without_error_reasons___no_error_section(self, capsys):
        """Test that no error section appears when there are no errors."""
        # Arrange
        status = self._create_report_status(error_reasons=None, num_errors=0)

        # Act
        status.print()

        # Assert
        captured = capsys.readouterr()
        assert 'Error reasons:' not in captured.out
        assert '0 time series ran into an error' in captured.out
        assert '10 time series finished' in captured.out

    def test_print___with_empty_list_error_reasons___no_error_section(self, capsys):
        """Test that no error section appears for empty error list."""
        # Arrange
        status = self._create_report_status(error_reasons=[], num_errors=0)

        # Act
        status.print()

        # Assert
        captured = capsys.readouterr()
        assert 'Error reasons:' not in captured.out

    def test_print___with_print_error_reasons_false___no_error_section(self, capsys):
        """Test that error section is suppressed when print_error_reasons=False."""
        # Arrange
        error_reasons = [
            ErrorReason(
                status='Error',
                error_message='Test error',
                timeseries=['sales_01']
            )
        ]
        status = self._create_report_status(error_reasons=error_reasons, num_errors=1)

        # Act
        status.print(print_error_reasons=False)

        # Assert
        captured = capsys.readouterr()
        assert 'Error reasons:' not in captured.out
        assert 'Test error' not in captured.out
        assert '1 time series ran into an error' in captured.out

    def test_print___with_prerequisites_and_errors___prints_all_errors(self, capsys):
        """Test that errors from prerequisite reports are also printed."""
        # Arrange
        prerequisite_status = ReportStatus(
            id=ReportIdentifier(report_id=100, settings_id=200),
            description='Prerequisite Report',
            result_type='matcher',
            progress=ReportStatusProgress(requested=5, pending=0, finished=5),
            results=ReportStatusResults(successful=4, no_evaluation=0, error=1),
            error_reasons=[
                ErrorReason(
                    status='Error',
                    error_message='Matcher failed: correlation too low',
                    timeseries=['ts_01']
                )
            ]
        )

        main_status = self._create_report_status(
            error_reasons=[
                ErrorReason(
                    status='Error',
                    error_message='Forecast failed: model convergence issue',
                    timeseries=['ts_02']
                )
            ],
            num_errors=1
        )
        main_status.prerequisites = [prerequisite_status]

        # Act
        main_status.print()

        # Assert
        captured = capsys.readouterr()
        assert 'Prerequisite Report' in captured.out
        assert '[Error] Matcher failed: correlation too low' in captured.out
        assert 'ts_01' in captured.out
        assert 'Test Report' in captured.out
        assert '[Error] Forecast failed: model convergence issue' in captured.out
        assert 'ts_02' in captured.out
