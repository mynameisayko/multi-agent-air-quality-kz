from __future__ import annotations

import json
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AIR_PATH = PROJECT_ROOT / "data" / "external" / "airdata_almaty_hourly" / "pm25.csv.gz"
WEATHER_PATH = PROJECT_ROOT / "data" / "external" / "weather_almaty_hourly" / "open_meteo_almaty_hourly.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "tft_almaty_hourly_weather"
REPORT_DIR = PROJECT_ROOT / "reports" / "tft_almaty_hourly_weather"


def load_weather() -> pd.DataFrame:
    obj = json.loads(WEATHER_PATH.read_text(encoding="utf-8"))
    weather = pd.DataFrame(obj["hourly"]).rename(columns={"time": "timestamp"})
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather["wind_dir_sin"] = np.sin(np.deg2rad(weather["wind_direction_10m"]))
    weather["wind_dir_cos"] = np.cos(np.deg2rad(weather["wind_direction_10m"]))
    return weather


def build_hourly_dataframe() -> pd.DataFrame:
    df = pd.read_csv(AIR_PATH, parse_dates=["datetime_utc"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df[df["datetime_utc"] >= pd.Timestamp("2021-12-01", tz="UTC")].copy()
    df["timestamp"] = df["datetime_utc"].dt.floor("h").dt.tz_convert(None)

    grouped = (
        df.groupby("timestamp")
        .agg(
            pm25_value=("value_ugm3", "mean"),
            pm25_median=("value_ugm3", "median"),
            pm25_min=("value_ugm3", "min"),
            pm25_max=("value_ugm3", "max"),
            pm25_std=("value_ugm3", "std"),
            n_stations=("station_id", "nunique"),
            n_clusters=("cluster_id", "nunique"),
        )
        .reset_index()
    )

    full_index = pd.date_range(grouped["timestamp"].min(), grouped["timestamp"].max(), freq="h")
    grouped = grouped.set_index("timestamp").reindex(full_index).rename_axis("timestamp").reset_index()
    grouped["n_stations"] = grouped["n_stations"].fillna(0)
    grouped["n_clusters"] = grouped["n_clusters"].fillna(0)
    grouped["pm25_value"] = grouped["pm25_value"].interpolate(limit_direction="both")
    grouped["pm25_median"] = grouped["pm25_median"].interpolate(limit_direction="both")
    grouped["pm25_min"] = grouped["pm25_min"].interpolate(limit_direction="both")
    grouped["pm25_max"] = grouped["pm25_max"].interpolate(limit_direction="both")
    grouped["pm25_std"] = grouped["pm25_std"].fillna(0.0)

    weather = load_weather()
    grouped = grouped.merge(weather, on="timestamp", how="left")

    grouped["pm25_lag_1"] = grouped["pm25_value"].shift(1)
    grouped["pm25_lag_24"] = grouped["pm25_value"].shift(24)
    grouped["pm25_lag_168"] = grouped["pm25_value"].shift(168)
    grouped["pm25_roll6_mean"] = grouped["pm25_value"].shift(1).rolling(6).mean()
    grouped["pm25_roll24_mean"] = grouped["pm25_value"].shift(1).rolling(24).mean()
    grouped["pm25_roll24_std"] = grouped["pm25_value"].shift(1).rolling(24).std()
    grouped["pm25_roll168_mean"] = grouped["pm25_value"].shift(1).rolling(168).mean()

    grouped["city"] = "almaty"
    grouped["time_idx"] = np.arange(len(grouped))
    grouped["hour"] = grouped["timestamp"].dt.hour.astype(str)
    grouped["day_of_week"] = grouped["timestamp"].dt.dayofweek.astype(str)
    grouped["month"] = grouped["timestamp"].dt.month.astype(str)
    grouped["day_of_year"] = grouped["timestamp"].dt.dayofyear
    grouped["is_weekend"] = (grouped["timestamp"].dt.dayofweek >= 5).astype(int)
    grouped["heating_season"] = grouped["timestamp"].dt.month.isin([10, 11, 12, 1, 2, 3, 4]).astype(int)

    grouped = grouped.dropna().reset_index(drop=True)
    grouped["time_idx"] = np.arange(len(grouped))
    return grouped


def make_datasets(
    df: pd.DataFrame,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, int, int]:
    max_encoder_length = 24 * 7
    max_prediction_length = 24

    training_cutoff = df["time_idx"].max() - (24 * 14)
    validation_cutoff = df["time_idx"].max() - (24 * 7)

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="pm25_value",
        group_ids=["city"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["city"],
        time_varying_known_categoricals=["hour", "day_of_week", "month"],
        time_varying_known_reals=[
            "time_idx",
            "day_of_year",
            "is_weekend",
            "heating_season",
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m",
            "wind_dir_sin",
            "wind_dir_cos",
        ],
        time_varying_unknown_reals=[
            "pm25_value",
            "pm25_median",
            "pm25_min",
            "pm25_max",
            "pm25_std",
            "n_stations",
            "n_clusters",
            "pm25_lag_1",
            "pm25_lag_24",
            "pm25_lag_168",
            "pm25_roll6_mean",
            "pm25_roll24_mean",
            "pm25_roll24_std",
            "pm25_roll168_mean",
        ],
        target_normalizer=GroupNormalizer(groups=["city"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[df.time_idx <= validation_cutoff],
        predict=True,
        stop_randomization=True,
    )
    test = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict=True,
        stop_randomization=True,
    )
    return training, validation, test, training_cutoff, validation_cutoff


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return {"mae": mae, "rmse": rmse}


def main() -> None:
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_hourly_dataframe()
    save_dataframe(df, PROCESSED_DIR / "almaty_hourly_pm25_city_weather.csv")

    training, validation, test, training_cutoff, validation_cutoff = make_datasets(df)
    train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)
    test_loader = test.to_dataloader(train=False, batch_size=128, num_workers=0)

    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename="tft-hourly-weather-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=4, mode="min")
    logger = CSVLogger(save_dir=str(REPORT_DIR), name="lightning_logs")

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=20,
        attention_head_size=4,
        dropout=0.15,
        hidden_continuous_size=10,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=2,
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger,
        enable_model_summary=True,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_callback.best_model_path)
    raw_predictions = best_model.predict(test_loader, mode="raw", return_x=True)
    prediction_tensor = raw_predictions.output.prediction[..., 1]
    actual_tensor = raw_predictions.x["decoder_target"]

    predicted = prediction_tensor.detach().cpu().numpy().reshape(-1)
    actual = actual_tensor.detach().cpu().numpy().reshape(-1)

    forecast_horizon = len(predicted)
    forecast_timestamps = df["timestamp"].iloc[-forecast_horizon:].reset_index(drop=True)
    forecast_df = pd.DataFrame(
        {
            "timestamp": forecast_timestamps,
            "actual_pm25": actual,
            "predicted_pm25": predicted,
            "absolute_error": np.abs(actual - predicted),
        }
    )
    save_dataframe(forecast_df, REPORT_DIR / "test_forecast.csv")

    metrics = evaluate_predictions(actual, predicted)
    metrics.update(
        {
            "train_rows": int((df["time_idx"] <= training_cutoff).sum()),
            "full_rows": int(len(df)),
            "train_start": str(df["timestamp"].min()),
            "train_end": str(df.loc[df["time_idx"] <= training_cutoff, "timestamp"].iloc[-1]),
            "validation_end": str(df.loc[df["time_idx"] <= validation_cutoff, "timestamp"].iloc[-1]),
            "test_start": str(forecast_timestamps.iloc[0]),
            "test_end": str(forecast_timestamps.iloc[-1]),
            "best_checkpoint": str(Path(checkpoint_callback.best_model_path).resolve()),
            "weather_source": "Open-Meteo archive hourly API",
            "weather_leakage_note": "Backtest uses realized historical weather as known future covariates. Operational deployment should replace these with weather forecasts.",
        }
    )

    with (REPORT_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
