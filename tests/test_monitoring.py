from pipeline.config import MONITORING_REPORT_PATH
from pipeline.monitoring import monitor_pipeline


def test_monitoring_generates_report():
    report = monitor_pipeline()

    assert isinstance(report, dict)
    assert report["dataset_rows"] > 0
    assert report["missing_values"] == 0
    assert "selected_model" in report
    assert "f1_score" in report

    assert MONITORING_REPORT_PATH.exists()