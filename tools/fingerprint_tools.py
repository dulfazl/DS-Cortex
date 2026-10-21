"""Dataset fingerprint: a fixed-length numeric vector, normalized with fixed constants so it
stays comparable across datasets. Similarity is plain Euclidean distance."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tools.common import clip01, jsonable

FEATURES = ["log_rows", "log_features", "pct_categorical", "pct_missing", "pct_outliers", "is_classification",
            "is_regression", "imbalance", "mean_abs_skew", "mean_abs_kurtosis", "mean_abs_corr", "mean_vif"]


def dataset_fingerprint(df: pd.DataFrame, task_family: str, target: str | None, imbalance_ratio: float | None,
                        mean_abs_corr: float, mean_vif: float) -> dict:
    feats = [c for c in df.columns if c != target]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
    cat = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    outlier_cells = 0
    for c in num:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_cells += int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    n_num_cells = max(1, sum(df[c].notna().sum() for c in num))
    skews = [abs(float(df[c].skew())) for c in num if df[c].notna().sum() > 2]
    kurts = [abs(float(df[c].kurt())) for c in num if df[c].notna().sum() > 3]
    raw = {
        "log_rows": math.log10(max(1, len(df))) / 6,
        "log_features": math.log10(len(feats) + 1) / 3,
        "pct_categorical": len(cat) / max(1, len(feats)),
        "pct_missing": float(df[feats].isna().mean().mean()) if feats else 0.0,
        "pct_outliers": min(1.0, (outlier_cells / n_num_cells) / 0.2),
        "is_classification": 1.0 if task_family == "classification" else 0.0,
        "is_regression": 1.0 if task_family in ("regression", "time_series") else 0.0,
        "imbalance": min(1.0, math.log10(imbalance_ratio) / 2) if imbalance_ratio and imbalance_ratio > 1 else 0.0,
        "mean_abs_skew": min(1.0, (np.mean(skews) if skews else 0.0) / 5),
        "mean_abs_kurtosis": min(1.0, (np.mean(kurts) if kurts else 0.0) / 20),
        "mean_abs_corr": clip01(mean_abs_corr),
        "mean_vif": min(1.0, (mean_vif or 1.0) / 20),
    }
    vec = [round(clip01(raw[k]), 5) for k in FEATURES]
    return jsonable({"vector": vec, "features": FEATURES, "raw": raw})


def fingerprint_distance(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))
