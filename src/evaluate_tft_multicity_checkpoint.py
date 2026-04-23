from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from risk import RISK_LABELS, add_risk_columns
from train_tft_multicity_station import (
    MODEL_DIR,
    PROCESSED_DIR,
    REPORT_DIR,
    build_dataframe,
    decode_predictions,
    evaluate,
    export_results,
    make_datasets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--max-stations-per-city", type=int, default=5)
    parser.add_argument("--encoder-hours", type=int, default=168)
    parser.add_argument("--prediction-hours", type=int, default=24)
    parser.add_argument("--validation-days", type=int, default=14)
    parser.add_argument("--test-days", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--interpolation-limit", type=int, default=3)
    parser.add_argument("--loss", choices=["quantile", "mae", "rmse"], default="mae")
    parser.add_argument("--evaluation-horizon", type=int, default=24)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--export-dir", default=None)
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args.start, args.max_stations_per_city, args.interpolation_limit)
    processed_path = PROCESSED_DIR / "kz_multicity_station_hourly_pm25.csv"
    df.to_csv(processed_path, index=False)

    _, _, test, full_df, split_info = make_datasets(
        df,
        encoder_hours=args.encoder_hours,
        prediction_hours=args.prediction_hours,
        validation_days=args.validation_days,
        test_days=args.test_days,
    )
    test_loader = test.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)

    checkpoint_path = Path(args.checkpoint_path)
    best_model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))
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

    point_prediction_index = 1 if args.loss == "quantile" else 0
    forecast = decode_predictions(
        dataset=test,
        full_df=full_df,
        raw_predictions=raw_predictions,
        prediction_hours=args.prediction_hours,
        point_prediction_index=point_prediction_index,
        evaluation_horizon=args.evaluation_horizon,
    )
    forecast_path = REPORT_DIR / "test_forecast_with_risk.csv"
    forecast.to_csv(forecast_path, index=False)

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
            "processed_data": str(processed_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "risk_threshold_note": "safe <=12.0, moderate <=35.4, unhealthy <=150.4, hazardous >150.4 ug/m3",
            "split": split_info,
            "args": vars(args),
        }
    )
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    checkpoint_copy = MODEL_DIR / checkpoint_path.name
    if checkpoint_path.resolve() != checkpoint_copy.resolve():
        shutil.copy2(checkpoint_path, checkpoint_copy)

    if args.export_dir:
        export_path = Path(args.export_dir)
        export_results(export_path, REPORT_DIR, MODEL_DIR, processed_path)
        metrics["export_dir"] = str(export_path.resolve())
        (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        shutil.copy2(REPORT_DIR / "metrics.json", export_path / f"reports_{REPORT_DIR.name}" / "metrics.json")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
