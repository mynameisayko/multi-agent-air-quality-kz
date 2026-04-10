# Colab GPU Runbook: Multi-city Station-level TFT

Use this in Google Colab with GPU enabled.

## 1. Runtime

In Colab:

- `Runtime` -> `Change runtime type`
- Hardware accelerator: `T4 GPU` or better

## 2. Install dependencies

```bash
!pip install lightning pytorch-forecasting scikit-learn pandas numpy
```

## 3. Upload or clone project

Option A: upload the project folder to Google Drive and mount it.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Option B: clone from GitHub after you push the private repository.

```bash
!git clone <YOUR_PRIVATE_REPO_URL>
%cd multi-agent-air-quality-kz
```

## 4. Download multi-city PM2.5 data

```bash
!python src/download_airdata_pm25_multicity.py
```

## 5. Fast GPU baseline

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

## 6. Hyperparameter tuning candidates

Run these one by one and compare `reports/tft_kz_multicity_station/metrics.json`.

```bash
!python src/train_tft_multicity_station.py --start 2023-01-01 --max-stations-per-city 15 --max-epochs 30 --hidden-size 32 --attention-head-size 4 --dropout 0.1 --learning-rate 0.005 --batch-size 256
```

```bash
!python src/train_tft_multicity_station.py --start 2023-01-01 --max-stations-per-city 15 --max-epochs 30 --hidden-size 64 --attention-head-size 4 --dropout 0.15 --learning-rate 0.005 --batch-size 256
```

```bash
!python src/train_tft_multicity_station.py --start 2022-01-01 --max-stations-per-city 20 --max-epochs 40 --hidden-size 64 --attention-head-size 8 --dropout 0.2 --learning-rate 0.003 --batch-size 256
```

## 7. Main outputs

- `models/tft_kz_multicity_station/tft-kz-station-best.ckpt`
- `reports/tft_kz_multicity_station/metrics.json`
- `reports/tft_kz_multicity_station/test_forecast_with_risk.csv`
- `data/processed/kz_multicity_station_hourly_pm25.csv`

## 8. What to optimize

Primary regression metrics:

- `MAE`
- `RMSE`

Primary health-risk metrics:

- `risk_macro_f1`
- `unhealthy_recall`
- `unhealthy_f1`

For health-risk warning, recall is important because missing dangerous pollution episodes is worse than issuing a few false warnings.
