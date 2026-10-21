"""A deliberately small drift check: two-sample tests per feature against the training data.
No auto-retraining, no live monitoring, no performance tracking."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from tools.common import jsonable

ALERT_THRESHOLD = 0.3
P_THRESHOLD = 0.01
MIN_EFFECT = 0.1
LIMITS = ("This is a two-sample drift test against the training data. It does not retrain the model, "
          "monitor live traffic or track model performance.")


def drift_check(train_df: pd.DataFrame, new_df: pd.DataFrame, exclude: list[str] | None = None) -> dict:
    """KS test per continuous feature, chi-square per categorical feature, and the shift in the
    missing-value rate. Drift score = share of features with p < 0.01 and a meaningful effect."""
    exclude = set(exclude or [])
    rows = {}
    for c in train_df.columns:
        if c in exclude:
            continue
        if c not in new_df:
            rows[c] = {"type": "missing_in_new", "drifted": True, "detail": "column absent from the new batch"}
            continue
        a, b = train_df[c], new_df[c]
        miss_shift = float(abs(b.isna().mean() - a.isna().mean()))
        rec = {"missing_shift": miss_shift}
        if pd.api.types.is_numeric_dtype(a) and a.nunique() > 10:
            x, y = a.dropna().astype(float), pd.to_numeric(b, errors="coerce").dropna()
            if len(x) < 5 or len(y) < 5:
                continue
            res = st.ks_2samp(x, y)
            rec.update(type="continuous", test="ks", statistic=float(res.statistic), p_value=float(res.pvalue),
                       effect=float(res.statistic), mean_train=float(x.mean()), mean_new=float(y.mean()))
        else:
            ca, cb = a.dropna().astype(str).value_counts(), b.dropna().astype(str).value_counts()
            cats = sorted(set(ca.index) | set(cb.index))
            if len(cats) < 2:
                continue
            tab = np.array([[ca.get(k, 0) for k in cats], [cb.get(k, 0) for k in cats]]) + 0.5
            chi2, p, _, _ = st.chi2_contingency(tab)
            v = float(np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1))))
            rec.update(type="categorical", test="chi_square", statistic=float(chi2), p_value=float(p), effect=v)
        rec["drifted"] = bool((rec["p_value"] < P_THRESHOLD and rec["effect"] >= MIN_EFFECT) or miss_shift >= 0.1)
        rows[c] = rec
    n = len(rows)
    drifted = [c for c, r in rows.items() if r["drifted"]]
    score = len(drifted) / n if n else 0.0
    return jsonable({"features": rows, "drifted": drifted, "drift_score": score, "alert": score >= ALERT_THRESHOLD,
                     "threshold": ALERT_THRESHOLD, "n_train": len(train_df), "n_new": len(new_df), "limits": LIMITS})
