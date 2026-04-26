from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline.config import (
    METRICS_PATH,
    MONITORING_REPORT_PATH,
    PREDICTION_LOG_PATH,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
)


def load_processed_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DATA_PATH)

    print("Loaded processed SECOM dataset")
    print("Shape:", df.shape)

    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    print("\nData quality checks")
    print("Missing values:", total_missing)
    print("Duplicate rows:", duplicate_rows)

    return {
        "missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
    }


def check_target_distribution(df: pd.DataFrame) -> dict:
    distribution = (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2)

    pass_rate = float(distribution.get(0, 0))
    fail_rate = float(distribution.get(1, 0))

    print("\nTarget distribution (%)")
    print(distribution)

    return {
        "pass_rate_percent": pass_rate,
        "fail_rate_percent": fail_rate,
    }


def load_model_metrics() -> dict:
    metrics_df = pd.read_csv(METRICS_PATH)
    selected_model = metrics_df.sort_values(by="f1_score", ascending=False).iloc[0]

    print("\nSelected model metrics")
    print(selected_model[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]])

    return {
        "selected_model": selected_model["model"],
        "accuracy": float(selected_model["accuracy"]),
        "precision": float(selected_model["precision"]),
        "recall": float(selected_model["recall"]),
        "f1_score": float(selected_model["f1_score"]),
        "roc_auc": float(selected_model["roc_auc"]),
        "pr_auc": float(selected_model["pr_auc"]),
    }


def check_prediction_log() -> dict:
    if not Path(PREDICTION_LOG_PATH).exists():
        print("\nPrediction log not found yet")
        return {
            "logged_predictions": 0,
            "predicted_pass_rate_percent": None,
            "predicted_fail_rate_percent": None,
        }

    log_df = pd.read_csv(PREDICTION_LOG_PATH)
    prediction_distribution = (
        log_df["prediction"].value_counts(normalize=True) * 100
    ).round(2)

    predicted_pass_rate = float(prediction_distribution.get(0, 0))
    predicted_fail_rate = float(prediction_distribution.get(1, 0))

    print("\nPrediction log summary")
    print("Logged predictions:", len(log_df))
    print(prediction_distribution)

    return {
        "logged_predictions": int(len(log_df)),
        "predicted_pass_rate_percent": predicted_pass_rate,
        "predicted_fail_rate_percent": predicted_fail_rate,
    }


def save_monitoring_report(report: dict) -> None:
    Path(MONITORING_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame([report])
    report_df.to_csv(MONITORING_REPORT_PATH, index=False)

    print(f"\nMonitoring report saved to: {MONITORING_REPORT_PATH}")


def monitor_pipeline() -> dict:
    df = load_processed_data()

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
    }

    report.update(check_data_quality(df))
    report.update(check_target_distribution(df))
    report.update(load_model_metrics())
    report.update(check_prediction_log())

    save_monitoring_report(report)

    print("\nMonitoring complete")

    return report


if __name__ == "__main__":
    monitor_pipeline()