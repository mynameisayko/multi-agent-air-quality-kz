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
AIR_DIR = PROJECT_ROOT / "data" / "external" / "airdata_almaty_daily"
WEATHER_PATH = PROJECT_ROOT / "data" / "external" / "weather_almaty_daily" / "open_meteo_almaty_daily.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "tft_almaty_daily_weather"
REPORT_DIR = PROJECT_ROOT / "reports" / "tft_almaty_daily_weather"


def load_daily_series(parameter: str) -> pd.DataFrame:
    path = AIR_DIR / f"{parameter}.csv.gz"
    df = pd.read_csv(path, parse_dates=["date"])
    renamed = df.rename(
        columns={
            "value_ugm3": f"{parameter}_value",
            "median_value": f"{parameter}_median",
            "min_cluster": f"{parameter}_min_cluster",
            "max_cluster": f"{parameter}_max_cluster",
            "std_cluster": f"{parameter}_std_cluster",
            "n_clusters": f"{parameter}_n_clusters",
            "n_stations": f"{parameter}_n_stations",
        }
    )
    return renamed


def load_weather() -> pd.DataFrame:
    obj = json.loads(WEATHER_PATH.read_text(encoding="utf-8"))
    weather = pd.DataFrame(obj["daily"]).rename(columns={"time": "date"})
    weather["date"] = pd.to_datetime(weather["date"])
    weather["wind_dir_sin"] = np.sin(np.deg2rad(weather["wind_direction_10m_dominant"]))
    weather["wind_dir_cos"] = np.cos(np.deg2rad(weather["wind_direction_10m_dominant"]))
    return weather


def build_model_dataframe() -> pd.DataFrame:
    pm25 = load_daily_series("pm25")
    covariates = [load_daily_series(name) for name in ["pm10", "no2", "so2", "co"]]

    df = pm25.copy()
    for cov in covariates:
        df = df.merge(cov, on="date", how="left")

    weather = load_weather()
    df = df.merge(weather, on="date", how="left")

    covariate_cols = [col for col in df.columns if col != "date" and not col.startswith("pm25_")]
    df = df[df[covariate_cols].notna().all(axis=1)].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Historical features available from past observations only.
    df["pm25_lag_1"] = df["pm25_value"].shift(1)
    df["pm25_lag_7"] = df["pm25_value"].shift(7)
    df["pm25_roll7_mean"] = df["pm25_value"].shift(1).rolling(7).mean()
    df["pm25_roll14_mean"] = df["pm25_value"].shift(1).rolling(14).mean()
    df["pm25_roll7_std"] = df["pm25_value"].shift(1).rolling(7).std()

    df["city"] = "almaty"
    df["time_idx"] = np.arange(len(df))
    df["month"] = df["date"].dt.month.astype(str)
    df["day_of_week"] = df["date"].dt.dayofweek.astype(str)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    df["heating_season"] = df["date"].dt.month.isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
    df["is_cold_month"] = df["date"].dt.month.isin([11, 12, 1, 2]).astype(int)

    df = df.dropna().reset_index(drop=True)
    df["time_idx"] = np.arange(len(df))
    return df


def make_datasets(
    df: pd.DataFrame,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, int, int]:
    max_encoder_length = 120
    max_prediction_length = 30

    training_cutoff = df["time_idx"].max() - 60
    validation_cutoff = df["time_idx"].max() - 30

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
        time_varying_known_categoricals=["month", "day_of_week"],
        time_varying_known_reals=[
            "time_idx",
            "day_of_year",
            "week_of_year",
            "is_weekend",
            "heating_season",
            "is_cold_month",
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_mean",
            "wind_speed_10m_max",
            "surface_pressure_mean",
            "wind_dir_sin",
            "wind_dir_cos",
        ],
        time_varying_unknown_reals=[
            "pm25_value",
            "pm25_median",
            "pm25_min_cluster",
            "pm25_max_cluster",
            "pm25_std_cluster",
            "pm25_n_clusters",
            "pm25_n_stations",
            "pm10_value",
            "no2_value",
            "so2_value",
            "co_value",
            "pm25_lag_1",
            "pm25_lag_7",
            "pm25_roll7_mean",
            "pm25_roll14_mean",
            "pm25_roll7_std",
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

    df = build_model_dataframe()
    save_dataframe(df, PROCESSED_DIR / "almaty_daily_multivariate_weather.csv")

    training, validation, test, training_cutoff, validation_cutoff = make_datasets(df)
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
    test_loader = test.to_dataloader(train=False, batch_size=64, num_workers=0)

    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename="tft-weather-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    logger = CSVLogger(save_dir=str(REPORT_DIR), name="lightning_logs")

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=24,
        attention_head_size=4,
        dropout=0.15,
        hidden_continuous_size=12,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )

    trainer = Trainer(
        max_epochs=25,
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
    forecast_dates = df["date"].iloc[-forecast_horizon:].reset_index(drop=True)
    forecast_df = pd.DataFrame(
        {
            "date": forecast_dates,
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
            "train_start": str(df["date"].min().date()),
            "train_end": str(df.loc[df["time_idx"] <= training_cutoff, "date"].iloc[-1].date()),
            "validation_end": str(df.loc[df["time_idx"] <= validation_cutoff, "date"].iloc[-1].date()),
            "test_start": str(forecast_dates.iloc[0].date()),
            "test_end": str(forecast_dates.iloc[-1].date()),
            "best_checkpoint": str(Path(checkpoint_callback.best_model_path).resolve()),
            "weather_source": "Open-Meteo archive daily API",
            "weather_leakage_note": "Backtest uses realized historical weather as known future covariates. Operational deployment should replace these with weather forecasts.",
        }
    )

    with (REPORT_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
