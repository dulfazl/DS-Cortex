"""The Pipeline Critic's checks. Each returns executed numbers plus a boolean flag and the
threshold it was judged against, so every objection is backed by a tool result."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools import pipeline_core as pc
from tools.common import jsonable
from tools.sklearn_tools import bounded_metric

SUPERVISED = ("classification", "regression")


def _drop(metric: str, reference: float, value: float) -> float:
    """How much worse `value` is than `reference`: absolute for bounded metrics, relative otherwise."""
    gap = pc.oriented(metric, reference) - pc.oriented(metric, value)
    return gap if bounded_metric(metric) else gap / max(abs(reference), 1e-9)


def overfit_check(spec: dict, df_train: pd.DataFrame, task: dict, seed: int, fast: bool = False) -> dict:
    """Train-validation gap (fresh 5-fold CV on this seed), learning-curve slope and the feature-to-row ratio."""
    metric = task["metric"]
    out: dict = {"metric": metric}
    cv = pc.cv_evaluate(spec, df_train, task, seed, folds=5, repeats=1) if task["task_family"] in SUPERVISED else {}
    if task["task_family"] in SUPERVISED and cv.get("train_mean") is not None:
        out["train_mean"], out["val_mean"] = cv["train_mean"], cv["primary_mean"]
        out["gap"] = float(_drop(metric, cv["train_mean"], cv["primary_mean"]))
        out["gap_threshold"] = 0.15 if bounded_metric(metric) else 0.5
        out["gap_severe"] = 0.25 if bounded_metric(metric) else 1.0
        sizes = [0.5, 1.0] if fast else [0.4, 0.7, 1.0]
        info = pc.prepare_supervised(df_train, task)
        rng = np.random.default_rng(seed)
        curve = []
        for frac in sizes:
            vals = []
            for tr, te in pc.splitter(task, len(info["y"]), 3, 1, seed, info["y"], info["groups"]):
                sub = rng.choice(tr, max(10, int(len(tr) * frac)), replace=False) if frac < 1 else tr
                res, _, _ = pc._fit_predict_supervised(spec, task, info, sub, te, seed, [metric])
                vals.append(res["val"][metric])
            curve.append(float(np.nanmean(vals)))
        out["learning_curve"] = {"sizes": sizes, "val_scores": curve}
        out["slope"] = float((pc.oriented(metric, curve[-1]) - pc.oriented(metric, curve[-2])) / (sizes[-1] - sizes[-2]))
        n_feat = len(info["X"].columns)
        out["feature_row_ratio"] = n_feat / max(1, len(info["y"]))
        out["overfit"] = bool(out["gap"] > out["gap_threshold"])
        out["severity"] = "high" if out["gap"] > out["gap_severe"] else "medium" if out["overfit"] else None
    else:
        out["overfit"] = False
        out["severity"] = None
    return jsonable(out)


def leakage_probe(spec: dict, df_train: pd.DataFrame, task: dict, seed: int, baseline_mean: float,
                  cv_mean: float, top_feature: str | None = None) -> dict:
    """Shuffle the target and confirm the score falls to baseline; drop the top feature and see
    whether performance collapses."""
    metric = task["metric"]
    out: dict = {"metric": metric, "baseline_mean": baseline_mean, "cv_mean": cv_mean}
    if task["task_family"] not in SUPERVISED:
        return jsonable({**out, "applicable": False, "leak_suspected": False})
    rng = np.random.default_rng(seed)
    shuffled = df_train.copy()
    shuffled[task["target"]] = rng.permutation(shuffled[task["target"]].values)
    sh = pc.cv_evaluate(spec, shuffled, task, seed, folds=3, repeats=1)
    out["shuffled_mean"] = sh["primary_mean"]
    margin = 0.05 if bounded_metric(metric) else 0.1
    out["shuffle_margin"] = margin
    out["shuffled_above_baseline"] = bool(_drop(metric, sh["primary_mean"], baseline_mean) > margin)
    gain = pc.oriented(metric, cv_mean) - pc.oriented(metric, baseline_mean)
    if top_feature and top_feature in df_train.columns and gain > 0:
        dropped = pc.cv_evaluate(spec, df_train.drop(columns=[top_feature]), task, seed, folds=3, repeats=1)
        out["top_feature"] = top_feature
        out["without_top_mean"] = dropped["primary_mean"]
        out["share_of_gain_from_top"] = float((pc.oriented(metric, cv_mean) - pc.oriented(metric, dropped["primary_mean"])) / gain)
        out["collapse"] = bool(out["share_of_gain_from_top"] > 0.8)
    else:
        out["collapse"] = False
    out["leak_suspected"] = bool(out["shuffled_above_baseline"] or out["collapse"])
    out["applicable"] = True
    return jsonable(out)


def subgroup_check(spec: dict, df_train: pd.DataFrame, task: dict, seed: int, segment_col: str | None = None) -> dict:
    """Metric per segment of the largest categorical column; flags any segment with a large drop."""
    metric = task["metric"]
    if task["task_family"] not in SUPERVISED:
        return {"applicable": False, "flag": False}
    info = pc.prepare_supervised(df_train, task)
    X = info["X"]
    if segment_col is None:
        cands = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and 2 <= X[c].nunique() <= 12]
        if not cands:
            return {"applicable": False, "flag": False, "reason": "no categorical column with 2 to 12 levels"}
        segment_col = max(cands, key=lambda c: X[c].nunique())
    n = len(info["y"])
    oof_pred = np.empty(n, dtype=float)
    oof_proba = np.zeros((n, max(2, len(info["classes"])))) if task["task_family"] == "classification" else None
    for tr, te in pc.splitter(task, n, 5, 1, seed, info["y"], info["groups"]):
        pipe = pc.make_pipeline(spec, task, info["cols"], seed, len(info["classes"]) or 2, info["pos_weight"])
        pipe.fit(X.iloc[tr], info["y"][tr])
        oof_pred[te] = pipe.predict(X.iloc[te])
        if oof_proba is not None:
            p = pipe.predict_proba(X.iloc[te])
            oof_proba[np.ix_(te, pipe.classes_)] = p
    overall = pc.supervised_scores([metric], info["y"], oof_pred.astype(info["y"].dtype), oof_proba, info["classes"], info["pos"])[metric]
    segs = {}
    seg_vals = X[segment_col].astype(str).values
    for level in np.unique(seg_vals):
        idx = np.where(seg_vals == level)[0]
        if len(idx) < 30:
            continue
        yy = info["y"][idx]
        if task["task_family"] == "classification" and len(np.unique(yy)) < 2:
            continue
        v = pc.supervised_scores([metric], yy, oof_pred[idx].astype(info["y"].dtype),
                                 None if oof_proba is None else oof_proba[idx], info["classes"], info["pos"])[metric]
        segs[level] = {"n": int(len(idx)), "score": v, "drop": float(_drop(metric, overall, v))}
    threshold = 0.1 if bounded_metric(metric) else 0.25
    worst = max(segs.items(), key=lambda kv: kv[1]["drop"]) if segs else None
    return jsonable({"applicable": True, "segment_column": segment_col, "overall": overall, "segments": segs,
                     "threshold": threshold, "worst_segment": worst[0] if worst else None,
                     "worst_drop": worst[1]["drop"] if worst else 0.0,
                     "flag": bool(worst and worst[1]["drop"] > threshold)})


def seed_stability(spec: dict, df_train: pd.DataFrame, task: dict, seed: int, n_seeds: int = 3,
                   fold_std: float | None = None) -> dict:
    """Rerun CV with three seeds and flag high variance of the mean score. The limit grows with the
    fold-to-fold spread (fold sd / sqrt(5)), so small datasets are not flagged for noise they cannot avoid."""
    metric = task["metric"]
    runs = [pc.cv_evaluate(spec, df_train, task, seed + i, folds=5 if task["task_family"] != "time_series" else 4,
                           repeats=1) for i in range(n_seeds)]
    means = [r["primary_mean"] for r in runs]
    sd = float(np.std(means))
    centre = max(abs(np.mean(means)), 1e-9)
    rel = sd if bounded_metric(metric) else sd / centre
    fsd = fold_std if fold_std is not None else float(np.mean([r["primary_std"] for r in runs]))
    noise = fsd / np.sqrt(5) if bounded_metric(metric) else fsd / np.sqrt(5) / centre
    threshold = max(0.02 if bounded_metric(metric) else 0.05, float(noise))
    return jsonable({"seeds": [seed + i for i in range(n_seeds)], "means": means, "sd": sd, "relative_sd": rel,
                     "threshold": threshold, "unstable": bool(rel > threshold)})


def shift_sensitivity(spec: dict, df_train: pd.DataFrame, task: dict, seed: int, cv_mean: float,
                      feature: str | None = None) -> dict:
    """Train on the lower 80 percent of one feature, validate on the top 20 percent (covariate shift)."""
    metric = task["metric"]
    if task["task_family"] not in SUPERVISED:
        return {"applicable": False, "flag": False}
    info = pc.prepare_supervised(df_train, task)
    X = info["X"]
    nums = [c for c in info["cols"]["numeric"] if X[c].nunique() > 10]
    if feature not in nums:
        feature = max(nums, key=lambda c: X[c].std() / (abs(X[c].mean()) + 1e-9)) if nums else None
    if feature is None:
        return {"applicable": False, "flag": False, "reason": "no continuous feature"}
    cut = X[feature].quantile(0.8)
    te = np.where(X[feature] > cut)[0]
    tr = np.where(~(X[feature] > cut))[0]
    if len(te) < 20 or (task["task_family"] == "classification" and len(np.unique(info["y"][te])) < 2):
        return {"applicable": False, "flag": False, "reason": "shifted slice too small"}
    res, _, _ = pc._fit_predict_supervised(spec, task, info, tr, te, seed, [metric])
    shifted = res["val"][metric]
    drop = float(_drop(metric, cv_mean, shifted))
    threshold = 0.1 if bounded_metric(metric) else 0.25
    return jsonable({"applicable": True, "feature": feature, "cut": float(cut), "shifted_score": shifted,
                     "cv_mean": cv_mean, "drop": drop, "threshold": threshold, "flag": bool(drop > threshold)})


def unnecessary_step_check(spec: dict) -> dict:
    """Steps that do nothing for this model, e.g. scaling before a tree model."""
    model, prep = spec["model"], spec.get("prep", {})
    found = []
    if model in pc.TREE_MODELS:
        if prep.get("scale"):
            found.append("scaling before a tree model")
        if prep.get("power"):
            found.append("power transform before a tree model")
        if prep.get("winsorize"):
            found.append("outlier capping before a tree model (trees split on ranks)")
    return {"model": model, "unnecessary": found, "flag": bool(found)}


def baseline_check(cv: dict, baseline_mean: float, metric: str) -> dict:
    gain = pc.oriented(metric, cv["primary_mean"]) - pc.oriented(metric, baseline_mean)
    return jsonable({"cv_mean": cv["primary_mean"], "baseline_mean": baseline_mean, "gain": gain,
                     "cv_std": cv["primary_std"], "beats_baseline": bool(gain > cv["primary_std"])})


def overlap_check(df_train: pd.DataFrame, task: dict) -> dict:
    """Offline RL: tiny logging propensities make IPW estimates explode."""
    p_col = task.get("propensity_column")
    if not p_col or p_col not in df_train:
        return {"applicable": False, "flag": False}
    p = pd.to_numeric(df_train[p_col], errors="coerce").dropna()
    share_small = float((p < 0.05).mean())
    return jsonable({"applicable": True, "min_propensity": float(p.min()), "share_below_0.05": share_small,
                     "max_weight": float(1 / max(p.min(), 1e-6)), "flag": bool(share_small > 0.05)})
