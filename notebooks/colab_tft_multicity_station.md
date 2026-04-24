# Colab GPU Runbook: Multi-city Station-level TFT

Use these cells in Google Colab. The key rule is simple: save to Google Drive, not to `/content` and not to `/kaggle/working`.

## 1. Enable GPU

- `Runtime` -> `Change runtime type`
- Hardware accelerator: `T4 GPU` or better

## 2. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 3. Clone or update the repo

First run:

```bash
%cd /content
!git clone https://github.com/mynameisayko/multi-agent-air-quality-kz.git
%cd /content/multi-agent-air-quality-kz
```

Later runs:

```bash
%cd /content/multi-agent-air-quality-kz
!git pull
```

## 4. Install dependencies

```bash
%cd /content/multi-agent-air-quality-kz
!pip install -r requirements.txt
!python -c "import pytorch_forecasting, lightning, torch; print('deps ok')"
!nvidia-smi
```

## 5. Download data

```bash
%cd /content/multi-agent-air-quality-kz
!python src/download_airdata_pm25_multicity.py
```

## 6. Safe Colab training run

If Drive is mounted, `train_tft_multicity_station.py` will automatically use `/content/drive/MyDrive/air-quality-kz-results` even if you omit `--export-dir`.

```bash
%cd /content/multi-agent-air-quality-kz
!python src/train_tft_multicity_station.py \
  --start 2024-01-01 \
  --max-stations-per-city 5 \
  --encoder-hours 168 \
  --prediction-hours 24 \
  --validation-days 14 \
  --test-days 14 \
  --evaluation-horizon 24 \
  --max-epochs 20 \
  --hidden-size 32 \
  --attention-head-size 4 \
  --dropout 0.1 \
  --learning-rate 0.003 \
  --batch-size 64 \
  --interpolation-limit 3 \
  --loss mae
```

## 7. Resume after session interruption

```bash
%cd /content/multi-agent-air-quality-kz
!python src/train_tft_multicity_station.py \
  --start 2024-01-01 \
  --max-stations-per-city 5 \
  --encoder-hours 168 \
  --prediction-hours 24 \
  --validation-days 14 \
  --test-days 14 \
  --evaluation-horizon 24 \
  --max-epochs 20 \
  --hidden-size 32 \
  --attention-head-size 4 \
  --dropout 0.1 \
  --learning-rate 0.003 \
  --batch-size 64 \
  --interpolation-limit 3 \
  --loss mae \
  --resume-from-checkpoint /content/drive/MyDrive/air-quality-kz-results/models_tft_kz_multicity_station/last.ckpt
```

## 8. Check what was saved

```bash
!find /content/drive/MyDrive/air-quality-kz-results -maxdepth 3 -type f | sort
```

Expected important files:

- `models_tft_kz_multicity_station/last.ckpt`
- `models_tft_kz_multicity_station/tft-kz-station-best.ckpt`
- `reports_tft_kz_multicity_station/metrics.json`
- `reports_tft_kz_multicity_station/run_state.json`
- `reports_tft_kz_multicity_station/test_forecast_with_risk.csv`

## 9. Compare with XGBoost

```bash
%cd /content/multi-agent-air-quality-kz
!python src/train_xgboost_baseline.py \
  --processed /content/drive/MyDrive/air-quality-kz-results/kz_multicity_station_hourly_pm25.csv \
  --tft-forecast /content/drive/MyDrive/air-quality-kz-results/reports_tft_kz_multicity_station/test_forecast_with_risk.csv \
  --output-dir /content/drive/MyDrive/air-quality-kz-results/xgboost_baseline
```

## 10. What to optimize

Primary regression metrics:

- `MAE`
- `RMSE`

Primary risk metrics:

- `risk_macro_f1`
- `unhealthy_recall`
- `unhealthy_f1`
