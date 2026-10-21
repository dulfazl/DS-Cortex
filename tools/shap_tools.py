"""Explainability tools. SHAP for tree and linear models; permutation importance as the fallback.
If SHAP fails for any reason, the substitution is recorded, never hidden."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools.common import jsonable

SCORERS = {"f1_macro": "f1_macro", "roc_auc": "roc_auc", "pr_auc": "average_precision", "rmse": "neg_root_mean_squared_error",
           "mae": "neg_mean_absolute_error", "r2": "r2", "recall_at_precision": "f1_macro"}


def _original_name(fname: str, cols: list[str]) -> str:
    for prefix in ("missingindicator_",):
        if fname.startswith(prefix):
            fname = fname[len(prefix):]
    best = None
    for c in cols:
        if fname == c or fname.startswith(c + "_"):
            if best is None or len(c) > len(best):
                best = c
    return best or fname


def _aggregate(values: np.ndarray, names: list[str], cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for v, n in zip(values, names):
        o = _original_name(n, cols)
        out[o] = out.get(o, 0.0) + float(v)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def permutation_importance(pipe, X: pd.DataFrame, y, metric: str, seed: int, n_repeats: int = 5) -> dict:
    from sklearn.inspection import permutation_importance as pi
    scoring = SCORERS.get(metric, None)
    if metric == "roc_auc" and len(np.unique(y)) > 2:
        scoring = "roc_auc_ovr"
    r = pi(pipe, X, y, n_repeats=n_repeats, random_state=seed, scoring=scoring, n_jobs=1)
    imp = dict(sorted({c: float(m) for c, m in zip(X.columns, r.importances_mean)}.items(), key=lambda kv: -kv[1]))
    return {"method": "permutation_importance", "importance": imp,
            "std": {c: float(s) for c, s in zip(X.columns, r.importances_std)}, "scoring": scoring}


def shap_summary(pipe, X: pd.DataFrame, y, task: dict, seed: int, max_rows: int = 400, force_fail: bool = False) -> dict:
    """Mean |SHAP| per original column. Falls back to permutation importance and says so."""
    X_s = X.sample(min(len(X), max_rows), random_state=seed) if len(X) > max_rows else X
    model_name = None
    try:
        if force_fail:
            raise RuntimeError("forced SHAP failure (test)")
        import shap
        prep, est = pipe.named_steps["prep"], pipe.named_steps["model"]
        Xt = np.asarray(prep.transform(X_s), dtype=float)
        names = list(prep.get_feature_names_out())
        model_name = type(est).__name__
        if any(k in model_name for k in ("Forest", "Boost", "XGB", "LGBM", "Tree")):
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(Xt, check_additivity=False)
            method = "shap_tree"
        elif any(k in model_name for k in ("Logistic", "Ridge", "Lasso", "Linear")):
            explainer = shap.LinearExplainer(est, Xt)
            sv = explainer.shap_values(Xt)
            method = "shap_linear"
        else:
            raise RuntimeError(f"no fast SHAP explainer for {model_name}")
        arr = np.abs(np.asarray(sv if not isinstance(sv, list) else np.stack(sv, axis=-1), dtype=float))
        while arr.ndim > 2:
            arr = arr.mean(axis=-1)
        per_feature = arr.mean(axis=0)
        agg = _aggregate(per_feature, names, list(X.columns))
        return jsonable({"method": method, "importance": agg, "n_rows": len(X_s), "substituted": False,
                         "model": model_name})
    except Exception as e:  # noqa: BLE001 - any SHAP failure falls back and is recorded
        y_s = y[X.index.get_indexer(X_s.index)] if len(X_s) < len(X) else y
        res = permutation_importance(pipe, X_s, y_s, task["metric"], seed)
        return jsonable({**res, "n_rows": len(X_s), "substituted": True, "model": model_name,
                         "substitution_reason": f"SHAP failed: {type(e).__name__}: {str(e)[:200]}"})
