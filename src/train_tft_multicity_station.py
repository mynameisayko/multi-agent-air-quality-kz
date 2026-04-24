from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_forecasting.metrics as pf_metrics
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from risk import RISK_LABELS, add_risk_columns, pm25_risk_class


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "external" / "airdata_kz_hourly_pm25"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "tft_kz_multicity_station"
REPORT_DIR = PROJECT_ROOT / "reports" / "tft_kz_multicity_station"
COLAB_DEFAULT_EXPORT_DIR = Path("/content/drive/MyDrive/air-quality-kz-results")

CITY_FILES = {
    "almaty": DATA_DIR / "almaty" / "pm25.csv.gz",
    "astana": DATA_DIR / "astana" / "pm25.csv.gz",
    "karaganda": DATA_DIR / "karaganda" / "pm25.csv.gz",
    "rest_of_kz": DATA_DIR / "rest_of_kz" / "pm25.csv.gz",
}

CITY_TIMEZONES = {
    "almaty": "Asia/Almaty",
    "astana": "Asia/Almaty",
    "karaganda": "Asia/Almaty",
    "rest_of_kz": "Asia/Almaty",
}

KNOWN_CATEGORICALS = ["hour", "day_of_week", "month"]
KNOWN_REALS = [
    "time_idx",
    "day_of_year",
    "is_weekend",
    "heating_season",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]
UNKNOWN_REALS = [
    "pm25_value",
    "pm25_lag_1",
    "pm25_lag_24",
    "pm25_lag_168",
    "pm25_roll6_mean",
    "pm25_roll24_mean",
    "pm25_roll24_std",
    "pm25_roll168_mean",
]


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
    local_tz = CITY_TIMEZONES[city]
    raw["local_timestamp"] = raw["datetime_utc"].dt.tz_convert(local_tz).dt.tz_localize(None).dt.floor("h")
    raw["city"] = city
    raw["station_id"] = raw["station_id"].astype(str)
    raw["station_name"] = raw["station_name"].fillna("unknown").astype(str)

    counts = raw.groupby("station_id")["value_ugm3"].count().sort_values(ascending=False)
    keep_stations = counts.head(max_stations).index
    raw = raw[raw["station_id"].isin(keep_stations)].copy()

    grouped = (
        raw.groupby(["city", "station_id", "station_name", "lat", "lon", "timestamp", "local_timestamp"], dropna=False)
        .agg(pm25_value=("value_ugm3", "mean"))
        .reset_index()
    )
    return grouped


def regularize_station(group: pd.DataFrame, interpolation_limit: int) -> pd.DataFrame:
    group = group.sort_values("timestamp")
    index = pd.date_range(group["timestamp"].min(), group["timestamp"].max(), freq="h")
    meta_cols = ["city", "station_id", "station_name", "lat", "lon"]
    meta = group[meta_cols].iloc[0].to_dict()
    local_meta = group.groupby("timestamp")["local_timestamp"].first()
    hourly = group.groupby("timestamp", as_index=False).agg(pm25_value=("pm25_value", "mean"))
    regular = hourly.set_index("timestamp").reindex(index).rename_axis("timestamp").reset_index()
    for col, value in meta.items():
        regular[col] = value
    regular["local_timestamp"] = local_meta.reindex(index).to_numpy()
    regular["local_timestamp"] = regular["local_timestamp"].ffill().bfill()

    # Causal fill only: propagate recent past values forward, then drop remaining gaps.
    regular["pm25_value"] = regular["pm25_value"].ffill(limit=interpolation_limit)
    return regular.dropna(subset=["pm25_value", "local_timestamp"])


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

    local_ts = df["local_timestamp"]
    local_hour = local_ts.dt.hour
    local_dow = local_ts.dt.dayofweek
    df["hour"] = local_hour.astype(str)
    df["day_of_week"] = local_dow.astype(str)
    df["month"] = local_ts.dt.month.astype(str)
    df["day_of_year"] = local_ts.dt.dayofyear
    df["is_weekend"] = (local_dow >= 5).astype(int)
    df["heating_season"] = local_ts.dt.month.isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
    df["hour_sin"] = np.sin(2 * math.pi * local_hour / 24.0)
    df["hour_cos"] = np.cos(2 * math.pi * local_hour / 24.0)
    df["dow_sin"] = np.sin(2 * math.pi * local_dow / 7.0)
    df["dow_cos"] = np.cos(2 * math.pi * local_dow / 7.0)
    df["risk_class"] = pm25_risk_class(df["pm25_value"]).astype(str)

    df = df.dropna().reset_index(drop=True)
    df["time_idx"] = df.groupby("series_id").cumcount()
    return df


def make_datasets(
    df: pd.DataFrame,
    encoder_hours: int,
    prediction_hours: int,
    validation_days: int,
    test_days: int,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame, dict[str, int]]:
    if validation_days < 1 or test_days < 1:
        raise ValueError("validation_days and test_days must be positive integers.")

    df = df.copy()
    max_time = int(df["time_idx"].max())
    test_start_idx = max_time - (test_days * 24) + 1
    validation_start_idx = test_start_idx - (validation_days * 24)
    min_train_idx = encoder_hours + prediction_hours
    if validation_start_idx <= min_train_idx:
        raise ValueError("Not enough history for the requested encoder/validation/test windows.")

    training_df = df[df["time_idx"] < validation_start_idx].copy()

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
        time_varying_known_categoricals=KNOWN_CATEGORICALS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[df["time_idx"] < test_start_idx].copy(),
        min_prediction_idx=validation_start_idx,
        predict=False,
        stop_randomization=True,
    )
    test = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=test_start_idx,
        predict=False,
        stop_randomization=True,
    )
    split_info = {
        "validation_start_idx": validation_start_idx,
        "test_start_idx": test_start_idx,
        "max_time_idx": max_time,
    }
    return training, validation, test, df, split_info


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    denom = np.maximum(np.abs(actual), 1e-6)
    return {
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mape": float(np.mean(np.abs((actual - predicted) / denom)) * 100.0),
        "bias": float(np.mean(predicted - actual)),
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


def detect_runtime() -> str:
    if "google.colab" in __import__("sys").modules:
        return "colab"
    if Path("/kaggle/working").exists():
        return "kaggle"
    return "local"


def resolve_export_dir(export_dir_arg: str | None) -> Path | None:
    runtime = detect_runtime()
    if export_dir_arg:
        export_dir = Path(export_dir_arg)
        if runtime == "colab" and str(export_dir).startswith("/kaggle/"):
            raise ValueError(
                "Colab run detected, but --export-dir points to /kaggle/. "
                "Use a Google Drive path like /content/drive/MyDrive/air-quality-kz-results."
            )
        return export_dir

    if runtime == "colab" and COLAB_DEFAULT_EXPORT_DIR.parent.exists():
        return COLAB_DEFAULT_EXPORT_DIR
    return None


def write_run_state(report_dir: Path, payload: dict[str, object]) -> None:
    (report_dir / "run_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


class ExportArtifactsCallback(Callback):
    def __init__(self, export_dir: Path | None, report_dir: Path, model_dir: Path, processed_path: Path) -> None:
        super().__init__()
        self.export_dir = export_dir
        self.report_dir = report_dir
        self.model_dir = model_dir
        self.processed_path = processed_path

    def _sync(self, trainer: Trainer, reason: str) -> None:
        if self.export_dir is None:
            return
        export_results(self.export_dir, self.report_dir, self.model_dir, self.processed_path)
        state = {
            "reason": reason,
            "current_epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "best_model_path": str(getattr(trainer.checkpoint_callback, "best_model_path", "")),
            "last_model_path": str(getattr(trainer.checkpoint_callback, "last_model_path", "")),
        }
        write_run_state(self.report_dir, state)
        shutil.copy2(self.report_dir / "run_state.json", self.export_dir / f"reports_{self.report_dir.name}" / "run_state.json")

    def on_fit_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        self._sync(trainer, reason="fit_start")

    def on_validation_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        self._sync(trainer, reason="validation_end")

    def on_exception(self, trainer: Trainer, pl_module: torch.nn.Module, exception: BaseException) -> None:
        self._sync(trainer, reason=f"exception:{exception.__class__.__name__}")

    def on_fit_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        self._sync(trainer, reason="fit_end")


def make_loss(loss_name: str) -> torch.nn.Module:
    if loss_name == "quantile":
        return pf_metrics.QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    if loss_name == "mae":
        return pf_metrics.MAE()
    if loss_name == "rmse":
        return pf_metrics.RMSE()
    raise ValueError(f"Unsupported loss: {loss_name}")


def make_logging_metrics() -> torch.nn.ModuleList:
    return torch.nn.ModuleList([pf_metrics.MAE(), pf_metrics.RMSE(), pf_metrics.SMAPE()])


def decode_predictions(
    dataset: TimeSeriesDataSet,
    full_df: pd.DataFrame,
    raw_predictions,
    prediction_hours: int,
    point_prediction_index: int,
    evaluation_horizon: int,
) -> pd.DataFrame:
    predicted = raw_predictions.output.prediction[..., point_prediction_index].detach().cpu().numpy()
    actual = raw_predictions.x["decoder_target"].detach().cpu().numpy()
    decoder_time_idx = raw_predictions.x["decoder_time_idx"].detach().cpu().numpy()

    index_df = dataset.x_to_index(raw_predictions.x).reset_index(drop=True)
    repeated = index_df.loc[index_df.index.repeat(prediction_hours)].reset_index(drop=True)
    repeated["time_idx"] = decoder_time_idx.reshape(-1)
    repeated["horizon_step"] = np.tile(np.arange(1, prediction_hours + 1), len(index_df))
    repeated["origin_time_idx"] = repeated["time_idx"] - repeated["horizon_step"]

    forecast = repeated.merge(
        full_df[["series_id", "time_idx", "timestamp", "local_timestamp", "city", "station_id", "station_name", "lat", "lon"]],
        on=["series_id", "time_idx"],
        how="left",
        validate="many_to_one",
    )
    origin_lookup = full_df[["series_id", "time_idx", "timestamp"]].rename(
        columns={"time_idx": "origin_time_idx", "timestamp": "origin_timestamp"}
    )
    forecast = forecast.merge(
        origin_lookup,
        on=["series_id", "origin_time_idx"],
        how="left",
        validate="many_to_one",
    )
    forecast["actual_pm25"] = actual.reshape(-1)
    forecast["predicted_pm25"] = predicted.reshape(-1)
    forecast["absolute_error"] = np.abs(forecast["actual_pm25"] - forecast["predicted_pm25"])
    if evaluation_horizon > 0:
        forecast = forecast[forecast["horizon_step"] == evaluation_horizon].copy()
    forecast = forecast.sort_values(["series_id", "time_idx", "origin_time_idx"]).reset_index(drop=True)
    forecast = add_risk_columns(forecast)
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--max-stations-per-city", type=int, default=10)
    parser.add_argument("--encoder-hours", type=int, default=168)
    parser.add_argument("--prediction-hours", type=int, default=24)
    parser.add_argument("--validation-days", type=int, default=14)
    parser.add_argument("--test-days", type=int, default=14)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--attention-head-size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--interpolation-limit", type=int, default=3)
    parser.add_argument("--loss", choices=["quantile", "mae", "rmse"], default="mae")
    parser.add_argument("--evaluation-horizon", type=int, default=24)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Optional directory for copying reports, checkpoints, and processed data. Use a Google Drive path in Colab.",
    )
    args = parser.parse_args()
    export_dir = resolve_export_dir(args.export_dir)

    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args.start, args.max_stations_per_city, args.interpolation_limit)
    processed_path = PROCESSED_DIR / "kz_multicity_station_hourly_pm25.csv"
    df.to_csv(processed_path, index=False)
    write_run_state(
        REPORT_DIR,
        {
            "reason": "pre_fit",
            "processed_data": str(processed_path.resolve()),
            "args": vars(args) | {"resolved_export_dir": str(export_dir) if export_dir else None},
        },
    )
    if export_dir:
        export_results(export_dir, REPORT_DIR, MODEL_DIR, processed_path)
        shutil.copy2(REPORT_DIR / "run_state.json", export_dir / f"reports_{REPORT_DIR.name}" / "run_state.json")

    training, validation, test, full_df, split_info = make_datasets(
        df,
        encoder_hours=args.encoder_hours,
        prediction_hours=args.prediction_hours,
        validation_days=args.validation_days,
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
        save_last=True,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=6, mode="min")
    logger = CSVLogger(save_dir=str(REPORT_DIR), name="lightning_logs")
    export_callback = ExportArtifactsCallback(
        export_dir=export_dir,
        report_dir=REPORT_DIR,
        model_dir=MODEL_DIR,
        processed_path=processed_path,
    )

    loss = make_loss(args.loss)
    point_prediction_index = 1 if args.loss == "quantile" else 0
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        hidden_continuous_size=max(8, args.hidden_size // 2),
        loss=loss,
        logging_metrics=make_logging_metrics(),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.devices,
        gradient_clip_val=0.1,
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint_callback, export_callback],
        logger=logger,
        enable_model_summary=True,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint,
    )

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

    forecast = decode_predictions(
        dataset=test,
        full_df=full_df,
        raw_predictions=raw_predictions,
        prediction_hours=args.prediction_hours,
        point_prediction_index=point_prediction_index,
        evaluation_horizon=args.evaluation_horizon,
    )
    forecast.to_csv(REPORT_DIR / "test_forecast_with_risk.csv", index=False)

    actual = forecast["actual_pm25"].to_numpy()
    predicted = forecast["predicted_pm25"].to_numpy()
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
            "forecast_rows": int(len(forecast)),
            "forecast_start": str(forecast["timestamp"].min()),
            "forecast_end": str(forecast["timestamp"].max()),
            "processed_data": str(processed_path.resolve()),
            "best_checkpoint": str(Path(checkpoint_callback.best_model_path).resolve()),
            "risk_threshold_note": "safe <=12.0, moderate <=35.4, unhealthy <=150.4, hazardous >150.4 ug/m3",
            "split": split_info,
            "args": vars(args),
        }
    )
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if export_dir:
        export_results(export_dir, REPORT_DIR, MODEL_DIR, processed_path)
        metrics["export_dir"] = str(export_dir.resolve())
        (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        shutil.copy2(REPORT_DIR / "metrics.json", export_dir / f"reports_{REPORT_DIR.name}" / "metrics.json")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
