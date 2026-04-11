# Kaggle Results Workflow

Use this workflow after training or after checkpoint evaluation in Kaggle.

## 1. Verify files

```bash
%cd /kaggle/working/multi-agent-air-quality-kz
!find models -type f -name "*.ckpt" -ls
!find reports/tft_kz_multicity_station -maxdepth 3 -type f -ls
```

## 2. Save results into Kaggle output

```bash
%cd /kaggle/working
!zip -r air-quality-kz-results.zip air-quality-kz-results
```

Download `air-quality-kz-results.zip` from the Kaggle output panel.

## 3. Restore results locally

Unzip the Kaggle archive into the project root or copy these files manually:

- `air-quality-kz-results/reports_tft_kz_multicity_station/metrics.json`
- `air-quality-kz-results/reports_tft_kz_multicity_station/test_forecast_with_risk.csv`
- `air-quality-kz-results/kz_multicity_station_hourly_pm25.csv`

Copy them locally to:

- `reports/tft_kz_multicity_station/metrics.json`
- `reports/tft_kz_multicity_station/test_forecast_with_risk.csv`
- `data/processed/kz_multicity_station_hourly_pm25.csv`

Do not commit `.ckpt` files to GitHub unless they are small and explicitly needed.

## 4. Generate article outputs

```bash
python src/train_xgboost_baseline.py

python src/prepare_article_outputs.py ^
  --forecast reports/tft_kz_multicity_station/test_forecast_with_risk.csv ^
  --processed data/processed/kz_multicity_station_hourly_pm25.csv ^
  --metrics reports/tft_kz_multicity_station/metrics.json ^
  --model-comparison reports/xgboost_baseline/model_comparison.csv ^
  --comparison-overlap reports/xgboost_baseline/comparison_forecast_overlap.csv
```

Generated figures:

- `manuscript/overleaf-upload/figures/actual_vs_predicted_pm25.png`
- `manuscript/overleaf-upload/figures/error_distribution.png`
- `manuscript/overleaf-upload/figures/risk_confusion_matrix.png`
- `manuscript/overleaf-upload/figures/baseline_model_comparison.png`
- `manuscript/overleaf-upload/figures/risk_class_distribution.png`

Generated tables:

- `reports/article_tables/baseline_comparison.csv`
- `reports/article_tables/forecast_with_baselines.csv`
