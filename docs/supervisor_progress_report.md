# Progress Report: MAS-TFT Air Quality Forecasting Project

## Project topic

**Multi-Agent System with Temporal Fusion Transformer for Urban Air Quality Forecasting and Health Risk Assessment in Kazakhstan**

The project develops a forecasting and decision-support pipeline for hourly PM2.5 prediction in Kazakhstan. The system uses public AirData.kz station-level observations, trains forecasting models, and converts PM2.5 forecasts into health-risk categories: **safe**, **moderate**, **unhealthy**, and **hazardous**.

## Current implementation status

The project currently has an end-to-end working pipeline:

1. Download and prepare hourly PM2.5 data from AirData.kz.
2. Build multi-city, station-level time series.
3. Train Temporal Fusion Transformer models.
4. Generate PM2.5 forecasts.
5. Convert PM2.5 predictions into health-risk classes.
6. Evaluate forecasting and risk-classification metrics.
7. Compare TFT with simple and machine-learning baselines.
8. Export results and figures for the manuscript.

## Models trained so far

Several TFT versions were tested during the pilot stage.

| Model | Scope | Main inputs | MAE | RMSE | Status |
|---|---|---|---:|---:|---|
| Daily TFT | Almaty city-level | Daily PM2.5 and pollutant covariates | 20.27 | 25.03 | Completed |
| Daily TFT + weather | Almaty city-level | Daily PM2.5, pollutants, weather | 19.18 | 22.73 | Completed |
| Hourly TFT | Almaty city-level | Hourly PM2.5, lag and rolling features | 16.60 | 20.87 | Completed |
| Hourly TFT + weather | Almaty city-level | Hourly PM2.5, weather, temporal features | 15.02 | 19.61 | Completed |
| Multi-city station TFT | Kazakhstan station-level | Hourly PM2.5, station metadata, temporal features | 11.62 | 17.62 | Preliminary |

The pilot experiments show that the TFT results improved when moving from daily to hourly data, when adding weather variables in the Almaty prototype, and when moving from a city-level aggregate to a multi-city station-level formulation.

## Baseline comparison

To make the evaluation more rigorous, three baseline models were added:

1. **Persistence 24h**: predicts that PM2.5 at `t+24` will be equal to PM2.5 at `t`.
2. **Rolling mean 24h**: predicts that PM2.5 at `t+24` will be equal to the mean PM2.5 during the previous 24 hours.
3. **XGBoost direct 24h**: a tabular machine-learning model that predicts PM2.5 at `t+24` using lag, rolling, calendar, station, city, and location features.

On the shared XGBoost-TFT test overlap, the current comparison is:

| Model | MAE | RMSE |
|---|---:|---:|
| TFT | 14.42 | 20.05 |
| XGBoost direct 24h | 8.05 | 11.37 |
| Persistence 24h | 7.97 | 12.65 |
| Rolling mean 24h | 10.77 | 14.72 |

The full exported TFT test result is MAE 11.62 and RMSE 17.62 on the complete TFT test file. The table above uses a smaller shared overlap where TFT, XGBoost, persistence, and rolling-mean predictions are all available for the same station-time pairs. Therefore, the TFT value in the comparison table is different from the full TFT metric.

This means that the current TFT checkpoint does not yet outperform XGBoost or persistence in MAE/RMSE. This is an important preliminary finding. It suggests that the TFT model is not fully tuned yet, while XGBoost is already strong because it directly uses engineered lag and rolling features.

## Health-risk classification results

The multi-city station-level TFT model also produces health-risk classes by mapping predicted PM2.5 concentrations to four categories:

| Risk class | PM2.5 threshold |
|---|---|
| Safe | <= 12.0 ug/m3 |
| Moderate | 12.1-35.4 ug/m3 |
| Unhealthy | 35.5-150.4 ug/m3 |
| Hazardous | > 150.4 ug/m3 |

Current TFT health-risk results:

| Metric | Value |
|---|---:|
| Risk accuracy | 0.674 |
| Risk macro F1 | 0.440 |
| Unhealthy precision | 1.000 |
| Unhealthy recall | 0.676 |
| Unhealthy F1 | 0.807 |

The risk layer is useful because it connects numerical PM2.5 forecasting with public-health warning categories. The current TFT model is conservative: when it predicts an unhealthy-or-worse event, it is usually correct, but it still misses some true unhealthy events.

## Why TFT is currently worse than baselines

The current result should be interpreted as a preliminary checkpoint, not as a final conclusion against TFT.

The main reasons are:

1. **Limited training and tuning**. TFT is a more complex neural architecture and requires longer training, early stopping, and hyperparameter search.
2. **Strong autocorrelation in PM2.5**. PM2.5 often remains similar from one day to the next, so persistence can be a very strong baseline.
3. **XGBoost benefits from engineered features**. XGBoost directly uses lag and rolling statistics, which are very informative for short-term PM2.5 prediction.
4. **Short test window**. The current evaluation uses a short test period, so the result may depend strongly on the selected days and stations.
5. **No full multi-city forecast weather yet**. Weather covariates improved the Almaty prototype, but the current multi-city TFT baseline does not yet include forecast meteorology for all stations.

## What to explain to the supervisor

The key message is:

> The project already has a working end-to-end pipeline for Kazakhstan PM2.5 forecasting and health-risk classification. Several TFT variants were trained, and the best TFT version is the multi-city station-level model. However, after adding stronger baselines, XGBoost and persistence currently perform better in MAE/RMSE. This shows that the current TFT checkpoint needs further tuning, more training, longer evaluation, and forecast weather covariates. The result is still valuable because it provides a transparent baseline comparison and a clear next-step research direction.

## Main code components

| File | Purpose |
|---|---|
| `src/download_airdata_pm25_multicity.py` | Downloads hourly PM2.5 data from AirData.kz for multiple Kazakhstani city/station groups. |
| `src/train_tft_multicity_station.py` | Trains the main multi-city station-level TFT model. |
| `src/evaluate_tft_multicity_checkpoint.py` | Evaluates a saved TFT checkpoint and exports metrics and forecasts. |
| `src/risk.py` | Converts PM2.5 values into health-risk classes. |
| `src/evaluate_risk.py` | Evaluates risk classification metrics. |
| `src/train_xgboost_baseline.py` | Trains the XGBoost direct 24-hour forecasting baseline. |
| `src/prepare_article_outputs.py` | Generates article tables and figures from forecast outputs. |
| `reports/tft_kz_multicity_station/metrics.json` | Stores TFT forecasting and risk metrics. |
| `reports/xgboost_baseline/metrics.json` | Stores XGBoost forecasting and risk metrics. |
| `reports/xgboost_baseline/model_comparison.csv` | Stores comparison between TFT, XGBoost, persistence, and rolling mean. |
| `manuscript/overleaf-upload/kz-air-quality-review.tex` | Main Overleaf-ready manuscript draft. |

## Next steps

The next technical steps are:

1. Tune TFT hyperparameters on GPU.
2. Train TFT for more epochs with early stopping.
3. Add forecast weather covariates for all cities and stations.
4. Extend the test window from 14 days to 30-60 days.
5. Compare all models on the same stations and same timestamps.
6. Add Random Forest or LightGBM as another classical ML baseline.
7. Improve unhealthy-event recall for the health-risk warning task.
8. Add TFT interpretability outputs, such as variable importance and attention analysis.

## Current conclusion

At the current stage, the project should be presented as a **preliminary but complete forecasting and health-risk assessment pipeline**. The strongest current numerical baseline is XGBoost/persistence, while TFT remains the main proposed architecture because it supports multi-horizon forecasting, station-level covariates, known future inputs, and interpretability. The next research step is not to abandon TFT, but to improve and tune it against the established baselines.
