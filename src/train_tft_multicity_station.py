from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from risk import RISK_LABELS, add_risk_columns, pm25_risk_class


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "external" / "airdata_kz_hourly_pm25"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "tft_kz_multicity_station"
REPORT_DIR = PROJECT_ROOT / "reports" / "tft_kz_multicity_station"

CITY_FILES = {
    "almaty": DATA_DIR / "almaty" / "pm25.csv.gz",
    "astana": DATA_DIR / "astana" / "pm25.csv.gz",
    "karaganda": DATA_DIR / "karaganda" / "pm25.csv.gz",
    "rest_of_kz": DATA_DIR / "rest_of_kz" / "pm25.csv.gz",
}


def load_city(city: str, path: Path, start: str, max_stations: int) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        parse_dates=["datetime_utc"],
        dtype={"station_id": str, "station_name": str, "cluster_id": str, "cluster_name": str},
        low_memory=False,
    )
    raw["datetime_utc"] = pd.to_datetime(raw["datetime_utc"], utc=True)
    raw = raw[raw["datetime_utc"] >= pd.Timestamp(start, tz="UTC")].copy()
    raw["timestamp"] = raw["datetime_utc"].dt.floor("h").dt.tz_convert(None)
    raw["city"] = city
    raw["station_id"] = raw["station_id"].astype(str)
    raw["station_name"] = raw["station_name"].fillna("unknown").astype(str)

    counts = raw.groupby("station_id")["value_ugm3"].count().sort_values(ascending=False)
    keep_stations = counts.head(max_stations).index
    raw = raw[raw["station_id"].isin(keep_stations)].copy()

    grouped = (
        raw.groupby(["city", "station_id", "station_name", "lat", "lon", "timestamp"], dropna=False)
        .agg(pm25_value=("value_ugm3", "mean"))
        .reset_index()
    )
    return grouped


def regularize_station(group: pd.DataFrame, interpolation_limit: int) -> pd.DataFrame:
    group = group.sort_values("timestamp")
    index = pd.date_range(group["timestamp"].min(), group["timestamp"].max(), freq="h")
    meta_cols = ["city", "station_id", "station_name", "lat", "lon"]
    meta = group[meta_cols].iloc[0].to_dict()
    hourly = group.groupby("timestamp", as_index=False).agg(pm25_value=("pm25_value", "mean"))
    regular = hourly.set_index("timestamp").reindex(index).rename_axis("timestamp").reset_index()
    for col, value in meta.items():
        regular[col] = value
    regular["pm25_value"] = regular["pm25_value"].interpolate(limit=interpolation_limit, limit_direction="both")
    return regular.dropna(subset=["pm25_value"])


def build_dataframe(start: str, max_stations_per_city: int, interpolation_limit: int) -> pd.DataFrame:
    frames = []
    for city, path in CITY_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run download_airdata_pm25_multicity.py first.")
        frames.append(load_city(city, path, start, max_stations_per_city))
    raw = pd.concat(frames, ignore_index=True)

    regularized = [
        regularize_station(group, interpolation_limit)
        for _, group in raw.groupby(["city", "station_id"], sort=False)
    ]
    df = pd.concat(regularized, ignore_index=True)
    df = df.sort_values(["city", "station_id", "timestamp"]).reset_index(drop=True)

    df["series_id"] = df["city"].astype(str) + "__" + df["station_id"].astype(str)
    df["city"] = df["city"].astype(str)
    df["station_id"] = df["station_id"].astype(str)
    df["pm25_lag_1"] = df.groupby("series_id")["pm25_value"].shift(1)
    df["pm25_lag_24"] = df.groupby("series_id")["pm25_value"].shift(24)
    df["pm25_lag_168"] = df.groupby("series_id")["pm25_value"].shift(168)
    df["pm25_roll6_mean"] = df.groupby("series_id")["pm25_value"].shift(1).rolling(6).mean().reset_index(level=0, drop=True)
    df["pm25_roll24_mean"] = df.groupby("series_id")["pm25_value"].shift(1).rolling(24).mean().reset_index(level=0, drop=True)
    df["pm25_roll24_std"] = df.groupby("series_id")["pm25_value"].shift(1).rolling(24).std().reset_index(level=0, drop=True)
    df["pm25_roll168_mean"] = df.groupby("series_id")["pm25_value"].shift(1).rolling(168).mean().reset_index(level=0, drop=True)

    df["hour"] = df["timestamp"].dt.hour.astype(str)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(str)
    df["month"] = df["timestamp"].dt.month.astype(str)
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    df["heating_season"] = df["timestamp"].dt.month.isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
    df["risk_class"] = pm25_risk_class(df["pm25_value"]).astype(str)

    df = df.dropna().reset_index(drop=True)
    df["time_idx"] = df.groupby("series_id").cumcount()
    return df


def make_datasets(
    df: pd.DataFrame,
    encoder_hours: int,
    prediction_hours: int,
    test_days: int,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame, pd.DataFrame]:
    max_time = df.groupby("series_id")["time_idx"].transform("max")
    df = df.copy()
    df["is_test"] = df["time_idx"] > (max_time - prediction_hours)
    df["is_validation_tail"] = df["time_idx"] > (max_time - test_days * 24)

    training_df = df[~df["is_validation_tail"]].copy()
    validation_df = df[~df["is_test"]].copy()

    training = TimeSeriesDataSet(
        training_df,
        time_idx="time_idx",
        target="pm25_value",
        group_ids=["series_id"],
        min_encoder_length=encoder_hours // 2,
        max_encoder_length=encoder_hours,
        min_prediction_length=prediction_hours,
        max_prediction_length=prediction_hours,
        static_categoricals=["city", "station_id"],
        static_reals=["lat", "lon"],
        time_varying_known_categoricals=["hour", "day_of_week", "month"],
        time_varying_known_reals=["time_idx", "day_of_year", "is_weekend", "heating_season"],
        time_varying_unknown_reals=[
            "pm25_value",
            "pm25_lag_1",
            "pm25_lag_24",
            "pm25_lag_168",
            "pm25_roll6_mean",
            "pm25_roll24_mean",
            "pm25_roll24_std",
            "pm25_roll168_mean",
        ],
        target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(training, validation_df, predict=True, stop_randomization=True)
    test = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    return training, validation, test, df, validation_df


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
    }


def export_results(export_dir: Path, report_dir: Path, model_dir: Path, processed_path: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        report_dir: export_dir / f"reports_{report_dir.name}",
        model_dir: export_dir / f"models_{model_dir.name}",
    }
    for source, destination in destinations.items():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

    data_destination = export_dir / processed_path.name
    shutil.copy2(processed_path, data_destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--max-stations-per-city", type=int, default=10)
    parser.add_argument("--encoder-hours", type=int, default=168)
    parser.add_argument("--prediction-hours", type=int, default=24)
    parser.add_argument("--test-days", type=int, default=14)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--attention-head-size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--interpolation-limit", type=int, default=3)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Optional directory for copying reports, checkpoints, and processed data. Use a Google Drive path in Colab.",
    )
    args = parser.parse_args()

    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args.start, args.max_stations_per_city, args.interpolation_limit)
    processed_path = PROCESSED_DIR / "kz_multicity_station_hourly_pm25.csv"
    df.to_csv(processed_path, index=False)

    training, validation, test, full_df, _ = make_datasets(
        df,
        encoder_hours=args.encoder_hours,
        prediction_hours=args.prediction_hours,
        test_days=args.test_days,
    )

    train_loader = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)
    test_loader = test.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)

    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename="tft-kz-station-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    logger = CSVLogger(save_dir=str(REPORT_DIR), name="lightning_logs")

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        hidden_continuous_size=max(8, args.hidden_size // 2),
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.devices,
        gradient_clip_val=0.1,
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger,
        enable_model_summary=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_callback.best_model_path)
    raw_predictions = best_model.predict(
        test_loader,
        mode="raw",
        return_x=True,
        trainer_kwargs={
            "accelerator": "auto",
            "devices": args.devices,
            "strategy": "auto",
        },
    )
    predicted = raw_predictions.output.prediction[..., 1].detach().cpu().numpy().reshape(-1)
    actual = raw_predictions.x["decoder_target"].detach().cpu().numpy().reshape(-1)

    # Predict=True returns the final prediction window for each station series.
    tails = (
        full_df.groupby("series_id", group_keys=False)
        .tail(args.prediction_hours)
        .sort_values(["series_id", "time_idx"])
        .reset_index(drop=True)
    )
    forecast = tails[["timestamp", "city", "station_id", "station_name", "lat", "lon"]].copy()
    forecast["actual_pm25"] = actual
    forecast["predicted_pm25"] = predicted
    forecast["absolute_error"] = np.abs(actual - predicted)
    forecast = add_risk_columns(forecast)
    forecast.to_csv(REPORT_DIR / "test_forecast_with_risk.csv", index=False)

    metrics = evaluate(actual, predicted)
    metrics.update(
        {
            "risk_accuracy": float(accuracy_score(forecast["actual_risk"], forecast["predicted_risk"])),
            "risk_macro_f1": float(f1_score(forecast["actual_risk"], forecast["predicted_risk"], labels=RISK_LABELS, average="macro", zero_division=0)),
            "unhealthy_precision": float(precision_score(forecast["actual_unhealthy_or_worse"], forecast["predicted_unhealthy_or_worse"], zero_division=0)),
            "unhealthy_recall": float(recall_score(forecast["actual_unhealthy_or_worse"], forecast["predicted_unhealthy_or_worse"], zero_division=0)),
            "unhealthy_f1": float(f1_score(forecast["actual_unhealthy_or_worse"], forecast["predicted_unhealthy_or_worse"], zero_division=0)),
            "series_count": int(full_df["series_id"].nunique()),
            "rows": int(len(full_df)),
            "processed_data": str(processed_path.resolve()),
            "best_checkpoint": str(Path(checkpoint_callback.best_model_path).resolve()),
            "risk_threshold_note": "safe <=12.0, moderate <=35.4, unhealthy <=150.4, hazardous >150.4 ug/m3",
            "args": vars(args),
        }
    )
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.export_dir:
        export_path = Path(args.export_dir)
        export_results(export_path, REPORT_DIR, MODEL_DIR, processed_path)
        metrics["export_dir"] = str(export_path.resolve())
        (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        shutil.copy2(REPORT_DIR / "metrics.json", export_path / f"reports_{REPORT_DIR.name}" / "metrics.json")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
