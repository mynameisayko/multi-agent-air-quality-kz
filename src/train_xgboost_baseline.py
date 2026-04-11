from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from risk import add_risk_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed" / "kz_multicity_station_hourly_pm25.csv"
DEFAULT_TFT_FORECAST = PROJECT_ROOT / "reports" / "tft_kz_multicity_station" / "test_forecast_with_risk.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "xgboost_baseline"


NUMERIC_FEATURES = [
    "pm25_value",
    "pm25_lag_1",
    "pm25_lag_24",
    "pm25_lag_168",
    "pm25_roll6_mean",
    "pm25_roll24_mean",
    "pm25_roll24_std",
    "pm25_roll168_mean",
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "heating_season",
    "lat",
    "lon",
]

CATEGORICAL_FEATURES = ["city", "station_id"]


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(np.mean(np.abs((actual - predicted) / np.clip(actual, 1e-6, None))) * 100.0),
    }


def warning_metrics(df: pd.DataFrame) -> dict[str, float]:
    unhealthy_labels = ["unhealthy", "hazardous"]
    actual_unhealthy = df["actual_risk"].isin(unhealthy_labels)
    predicted_unhealthy = df["predicted_risk"].isin(unhealthy_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_unhealthy,
        predicted_unhealthy,
        average="binary",
        zero_division=0,
    )
    return {
        "risk_accuracy": float(accuracy_score(df["actual_risk"], df["predicted_risk"])),
        "risk_macro_f1": float(f1_score(df["actual_risk"], df["predicted_risk"], average="macro", zero_division=0)),
        "unhealthy_precision": float(precision),
        "unhealthy_recall": float(recall),
        "unhealthy_f1": float(f1),
    }


def build_direct_24h_frame(processed_path: Path, horizon_hours: int) -> pd.DataFrame:
    df = pd.read_csv(
        processed_path,
        parse_dates=["timestamp"],
        dtype={"city": "string", "station_id": "string", "station_name": "string", "series_id": "string"},
    )
    df = df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    df["target_timestamp"] = df.groupby("series_id")["timestamp"].shift(-horizon_hours)
    df["target_pm25"] = df.groupby("series_id")["pm25_value"].shift(-horizon_hours)
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["series_id", "timestamp", "target_timestamp", "target_pm25"]
    return df[required].dropna().reset_index(drop=True)


def add_origin_keys(tft_forecast_path: Path, horizon_hours: int) -> pd.DataFrame:
    test = pd.read_csv(
        tft_forecast_path,
        parse_dates=["timestamp"],
        dtype={"city": "string", "station_id": "string", "station_name": "string"},
    )
    test["series_id"] = test["city"].astype(str) + "__" + test["station_id"].astype(str)
    test["origin_timestamp"] = test["timestamp"] - pd.Timedelta(hours=horizon_hours)
    return test


def make_model(args: argparse.Namespace) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        tree_method=args.tree_method,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct 24-hour XGBoost PM2.5 baseline.")
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--tft-forecast", type=Path, default=DEFAULT_TFT_FORECAST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--min-child-weight", type=float, default=3.0)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--tree-method", default="hist")
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_direct_24h_frame(args.processed, args.horizon_hours)
    tft_test = add_origin_keys(args.tft_forecast, args.horizon_hours)

    test_keys = tft_test[["series_id", "origin_timestamp", "timestamp"]].rename(
        columns={"origin_timestamp": "timestamp", "timestamp": "target_timestamp"}
    )
    test_frame = frame.merge(test_keys, on=["series_id", "timestamp", "target_timestamp"], how="inner")
    if test_frame.empty:
        raise ValueError("No matching XGBoost test rows found for the TFT forecast file.")

    min_test_origin_by_series = test_frame.groupby("series_id")["timestamp"].min()
    train_mask = frame.apply(
        lambda row: row["series_id"] in min_test_origin_by_series
        and row["target_timestamp"] < min_test_origin_by_series[row["series_id"]],
        axis=1,
    )
    train_frame = frame.loc[train_mask].copy()
    if train_frame.empty:
        raise ValueError("No training rows available before the TFT test origins.")

    pipeline = make_model(args)
    pipeline.fit(train_frame[CATEGORICAL_FEATURES + NUMERIC_FEATURES], train_frame["target_pm25"])

    predicted = np.clip(pipeline.predict(test_frame[CATEGORICAL_FEATURES + NUMERIC_FEATURES]), 0.0, None)
    out = test_frame[["timestamp", "target_timestamp", "series_id", "city", "station_id", "pm25_value", "pm25_roll24_mean", "target_pm25"]].copy()
    out = out.rename(columns={"timestamp": "origin_timestamp", "target_timestamp": "timestamp", "target_pm25": "actual_pm25"})
    out["predicted_pm25"] = predicted
    out["absolute_error"] = np.abs(out["actual_pm25"] - out["predicted_pm25"])
    out = add_risk_columns(out)

    xgb_forecast = out[["timestamp", "series_id", "city", "station_id", "actual_pm25", "predicted_pm25", "absolute_error", "actual_risk", "predicted_risk", "risk_match", "actual_unhealthy_or_worse", "predicted_unhealthy_or_worse"]].copy()
    overlap = xgb_forecast.merge(
        tft_test[["timestamp", "city", "station_id", "predicted_pm25"]].rename(columns={"predicted_pm25": "tft_predicted_pm25"}),
        on=["timestamp", "city", "station_id"],
        how="left",
    )
    overlap = overlap.merge(
        out[["timestamp", "series_id", "pm25_value", "pm25_roll24_mean"]].rename(
            columns={"pm25_value": "persistence_24h", "pm25_roll24_mean": "rolling_mean_24h"}
        ),
        on=["timestamp", "series_id"],
        how="left",
    ).dropna(subset=["tft_predicted_pm25", "persistence_24h", "rolling_mean_24h"])

    comparison_rows = [
        {"model": "TFT", **regression_metrics(overlap["actual_pm25"].to_numpy(), overlap["tft_predicted_pm25"].to_numpy())},
        {"model": "XGBoost direct 24h", **regression_metrics(overlap["actual_pm25"].to_numpy(), overlap["predicted_pm25"].to_numpy())},
        {"model": "Persistence 24h", **regression_metrics(overlap["actual_pm25"].to_numpy(), overlap["persistence_24h"].to_numpy())},
        {"model": "Rolling mean 24h", **regression_metrics(overlap["actual_pm25"].to_numpy(), overlap["rolling_mean_24h"].to_numpy())},
    ]
    comparison = pd.DataFrame(comparison_rows)

    metrics = {
        **regression_metrics(xgb_forecast["actual_pm25"].to_numpy(), xgb_forecast["predicted_pm25"].to_numpy()),
        **warning_metrics(xgb_forecast),
        "model": "XGBoost direct 24h",
        "horizon_hours": args.horizon_hours,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(xgb_forecast)),
        "comparison_rows": int(len(overlap)),
        "series_count": int(xgb_forecast["series_id"].nunique()),
        "args": vars(args) | {"processed": str(args.processed), "tft_forecast": str(args.tft_forecast), "output_dir": str(args.output_dir)},
    }

    xgb_forecast.to_csv(args.output_dir / "test_forecast_with_risk.csv", index=False)
    overlap.to_csv(args.output_dir / "comparison_forecast_overlap.csv", index=False)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
