from __future__ import annotations

import numpy as np
import pandas as pd


RISK_LABELS = ["safe", "moderate", "unhealthy", "hazardous"]


def pm25_risk_class(values: pd.Series | np.ndarray) -> pd.Categorical:
    """Map PM2.5 concentration to a compact four-level health-risk proxy.

    Thresholds follow a merged PM2.5 AQI-style interpretation:
    safe: <= 12.0 ug/m3
    moderate: 12.1-35.4 ug/m3
    unhealthy: 35.5-150.4 ug/m3
    hazardous: > 150.4 ug/m3

    This is a risk-warning target, not a clinical diagnosis. For operational
    deployment, the thresholds should be aligned with the target authority.
    """
    bins = [-np.inf, 12.0, 35.4, 150.4, np.inf]
    return pd.cut(values, bins=bins, labels=RISK_LABELS)


def add_risk_columns(
    df: pd.DataFrame,
    actual_col: str = "actual_pm25",
    predicted_col: str = "predicted_pm25",
) -> pd.DataFrame:
    out = df.copy()
    out["actual_risk"] = pm25_risk_class(out[actual_col]).astype(str)
    out["predicted_risk"] = pm25_risk_class(out[predicted_col]).astype(str)
    out["risk_match"] = out["actual_risk"] == out["predicted_risk"]
    out["actual_unhealthy_or_worse"] = out["actual_risk"].isin(["unhealthy", "hazardous"])
    out["predicted_unhealthy_or_worse"] = out["predicted_risk"].isin(["unhealthy", "hazardous"])
    return out

