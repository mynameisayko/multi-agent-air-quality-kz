from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from risk import RISK_LABELS, add_risk_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTICLE_FIGURES_DIR = PROJECT_ROOT / "manuscript" / "overleaf-upload" / "figures"
ARTICLE_TABLES_DIR = PROJECT_ROOT / "reports" / "article_tables"


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
    }


def build_baselines(processed: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    processed = processed.copy()
    processed["timestamp"] = pd.to_datetime(processed["timestamp"])
    forecast = forecast.copy()
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])

    processed = processed.sort_values(["series_id", "timestamp"])
    processed["persistence_24h"] = processed.groupby("series_id")["pm25_value"].shift(24)
    processed["rolling24_mean"] = (
        processed.groupby("series_id")["pm25_value"]
        .shift(1)
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    keys = ["timestamp", "city", "station_id"]
    baseline_cols = keys + ["persistence_24h", "rolling24_mean"]
    merged = forecast.merge(processed[baseline_cols], on=keys, how="left")
    return merged.dropna(subset=["persistence_24h", "rolling24_mean"])


def write_baseline_table(forecast: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    rows.append({"model": "TFT", **regression_metrics(forecast["actual_pm25"].to_numpy(), forecast["predicted_pm25"].to_numpy())})
    rows.append({"model": "Persistence 24h", **regression_metrics(forecast["actual_pm25"].to_numpy(), forecast["persistence_24h"].to_numpy())})
    rows.append({"model": "Rolling mean 24h", **regression_metrics(forecast["actual_pm25"].to_numpy(), forecast["rolling24_mean"].to_numpy())})
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "baseline_comparison.csv", index=False)
    return table


def plot_actual_vs_predicted(forecast: pd.DataFrame, out_dir: Path) -> None:
    sample_series = forecast["station_id"].value_counts().index[0]
    sample = forecast[forecast["station_id"] == sample_series].sort_values("timestamp")

    plt.figure(figsize=(10, 4))
    plt.plot(sample["timestamp"], sample["actual_pm25"], label="Actual PM2.5", linewidth=2)
    plt.plot(sample["timestamp"], sample["predicted_pm25"], label="Predicted PM2.5", linewidth=2)
    plt.title(f"Actual vs predicted PM2.5 for station {sample_series}")
    plt.xlabel("Timestamp")
    plt.ylabel("PM2.5, ug/m3")
    plt.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "actual_vs_predicted_pm25.png", dpi=200)
    plt.close()


def plot_error_distribution(forecast: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(forecast["absolute_error"], bins=30, color="#2f6f73", edgecolor="white")
    plt.title("Absolute error distribution")
    plt.xlabel("Absolute error, ug/m3")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "error_distribution.png", dpi=200)
    plt.close()


def plot_confusion_matrix(forecast: pd.DataFrame, out_dir: Path) -> None:
    labels = [label for label in RISK_LABELS if label in set(forecast["actual_risk"]) | set(forecast["predicted_risk"])]
    matrix = confusion_matrix(forecast["actual_risk"], forecast["predicted_risk"], labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Health-risk class confusion matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "risk_confusion_matrix.png", dpi=200)
    plt.close()


def plot_model_comparison(table: pd.DataFrame, out_dir: Path) -> None:
    x = np.arange(len(table))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, table["mae"], width, label="MAE")
    plt.bar(x + width / 2, table["rmse"], width, label="RMSE")
    plt.xticks(x, table["model"], rotation=20, ha="right")
    plt.ylabel("Error, ug/m3")
    plt.title("Forecasting model comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "baseline_model_comparison.png", dpi=200)
    plt.close()


def plot_risk_distribution(forecast: pd.DataFrame, out_dir: Path) -> None:
    counts = forecast["actual_risk"].value_counts().reindex(RISK_LABELS, fill_value=0)
    plt.figure(figsize=(7, 4))
    plt.bar(counts.index, counts.values, color="#7f4f24")
    plt.title("Actual PM2.5 health-risk class distribution")
    plt.xlabel("Risk class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "risk_class_distribution.png", dpi=200)
    plt.close()


def copy_metrics(metrics_path: Path, out_dir: Path) -> None:
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        (out_dir / "kaggle_multicity_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", required=True, help="Path to test_forecast_with_risk.csv")
    parser.add_argument("--processed", required=True, help="Path to kz_multicity_station_hourly_pm25.csv")
    parser.add_argument("--metrics", default=None, help="Optional path to metrics.json")
    parser.add_argument("--model-comparison", default=None, help="Optional model comparison CSV, for example XGBoost overlap results")
    args = parser.parse_args()

    ARTICLE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTICLE_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    forecast = pd.read_csv(args.forecast)
    if "actual_risk" not in forecast.columns or "predicted_risk" not in forecast.columns:
        forecast = add_risk_columns(forecast)
    processed = pd.read_csv(args.processed)

    forecast_with_baselines = build_baselines(processed, forecast)
    forecast_with_baselines.to_csv(ARTICLE_TABLES_DIR / "forecast_with_baselines.csv", index=False)

    if args.model_comparison:
        baseline_table = pd.read_csv(args.model_comparison)
        baseline_table.to_csv(ARTICLE_TABLES_DIR / "model_comparison.csv", index=False)
    else:
        baseline_table = write_baseline_table(forecast_with_baselines, ARTICLE_TABLES_DIR)
    plot_actual_vs_predicted(forecast_with_baselines, ARTICLE_FIGURES_DIR)
    plot_error_distribution(forecast_with_baselines, ARTICLE_FIGURES_DIR)
    plot_confusion_matrix(forecast_with_baselines, ARTICLE_FIGURES_DIR)
    plot_model_comparison(baseline_table, ARTICLE_FIGURES_DIR)
    plot_risk_distribution(forecast_with_baselines, ARTICLE_FIGURES_DIR)

    if args.metrics:
        copy_metrics(Path(args.metrics), ARTICLE_TABLES_DIR)

    print("Wrote article figures to", ARTICLE_FIGURES_DIR)
    print("Wrote article tables to", ARTICLE_TABLES_DIR)


if __name__ == "__main__":
    main()
