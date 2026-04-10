# Google Colab Quickstart

Use `T4 GPU`, not TPU, for this project.

## Setup

```bash
!git clone <YOUR_PRIVATE_REPO_URL>
%cd multi-agent-air-quality-kz
!pip install -r requirements.txt
```

## Download Kazakhstan PM2.5 data

```bash
!python src/download_airdata_pm25_multicity.py
```

## Run multi-city station-level TFT

```bash
!python src/train_tft_multicity_station.py \
  --start 2023-01-01 \
  --max-stations-per-city 10 \
  --max-epochs 20 \
  --hidden-size 32 \
  --attention-head-size 4 \
  --dropout 0.1 \
  --learning-rate 0.01 \
  --batch-size 256
```

## Outputs

- `reports/tft_kz_multicity_station/metrics.json`
- `reports/tft_kz_multicity_station/test_forecast_with_risk.csv`
- `models/tft_kz_multicity_station/tft-kz-station-best.ckpt`
