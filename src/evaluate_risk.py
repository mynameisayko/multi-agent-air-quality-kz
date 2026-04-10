from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from risk import RISK_LABELS, add_risk_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_forecast(report_dir: Path, timestamp_col: str) -> dict[str, float]:
    forecast_path = report_dir / "test_forecast.csv"
    forecast = pd.read_csv(forecast_path)
    enriched = add_risk_columns(forecast)
    enriched.to_csv(report_dir / "test_forecast_with_risk.csv", index=False)

    y_true = enriched["actual_risk"]
    y_pred = enriched["predicted_risk"]
    binary_true = enriched["actual_unhealthy_or_worse"]
    binary_pred = enriched["predicted_unhealthy_or_worse"]

    metrics = {
        "risk_accuracy": float(accuracy_score(y_true, y_pred)),
        "risk_macro_f1": float(f1_score(y_true, y_pred, labels=RISK_LABELS, average="macro", zero_division=0)),
        "unhealthy_precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
        "unhealthy_recall": float(recall_score(binary_true, binary_pred, zero_division=0)),
        "unhealthy_f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
    }

    report = classification_report(y_true, y_pred, labels=RISK_LABELS, zero_division=0)
    (report_dir / "risk_classification_report.txt").write_text(report, encoding="utf-8")

    existing_metrics_path = report_dir / "metrics.json"
    if existing_metrics_path.exists():
        existing = json.loads(existing_metrics_path.read_text(encoding="utf-8"))
    else:
        existing = {}
    existing.update(metrics)
    existing["risk_threshold_note"] = "safe <=12.0, moderate <=35.4, unhealthy <=150.4, hazardous >150.4 ug/m3"
    existing["risk_forecast_file"] = str((report_dir / "test_forecast_with_risk.csv").resolve())
    existing_metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    experiments = [
        (PROJECT_ROOT / "reports" / "tft_almaty_hourly", "timestamp"),
        (PROJECT_ROOT / "reports" / "tft_almaty_hourly_weather", "timestamp"),
        (PROJECT_ROOT / "reports" / "tft_almaty_daily", "date"),
        (PROJECT_ROOT / "reports" / "tft_almaty_daily_weather", "date"),
    ]
    summary = {}
    for report_dir, timestamp_col in experiments:
        if (report_dir / "test_forecast.csv").exists():
            summary[report_dir.name] = evaluate_forecast(report_dir, timestamp_col)
    output_path = PROJECT_ROOT / "reports" / "risk_metrics_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

