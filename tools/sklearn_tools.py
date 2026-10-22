"""Pipeline factory, evaluation and tuning. Candidates are declarative specs; every one becomes a
scikit-learn Pipeline (imputer, encoder, scaler, model), so all fitting happens inside CV."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tools import pipeline_core as pc
from tools.common import jsonable

LINEAR = {"logreg", "logreg_l1", "ridge", "lasso", "greedy_ridge"}


def _prep(signals: dict, model: str, plan: dict) -> dict:
    linear = model in LINEAR
    return {
        "impute_num": "median",
        "add_indicator": bool(signals.get("add_indicator")),
        "winsorize": bool(signals.get("winsorize")) and linear,
        "power": bool(signals.get("power_transform")) and linear,
        "scale": linear or model in ("ocsvm", "lof", "kmeans", "gmm", "dbscan", "agglomerative"),
        "high_card_encoding": "target" if plan["task_family"] in ("classification", "regression") else "frequency",
    }


def candidate_specs(plan: dict, signals: dict, n_rows: int, n_features: int, fast: bool = False,
                    cluster_choice: dict | None = None, exclude: tuple[str, ...] = ()) -> dict:
    """One list of candidate specs per task family, conditioned on upstream findings. `exclude` lists models
    this host should not fit (EXCLUDE_MODELS); at least two candidates are always kept."""
    fam = plan["task_family"]
    small = n_rows < 300 or signals.get("small_sample")
    n_est = 120 if fast else 250
    bal = "class_weight" if (signals.get("class_weight") or plan.get("imbalanced")) else None
    reasons: list[str] = []
    specs: list[dict] = []

    def add(model, params=None, why="", interpretable=None, favored=False):
        specs.append({"model": model, "params": params or {}, "prep": _prep(signals, model, plan),
                      "balance": bal if fam == "classification" else None, "reason": why,
                      "interpretable": model in pc.INTERPRETABLE if interpretable is None else interpretable,
                      "favored": favored})

    if small:
        reasons.append(f"only {n_rows} rows: tree depth and boosting rounds are capped")
    if signals.get("high_vif"):
        reasons.append("high VIF: regularized linear models are included and favored")
    if signals.get("power_transform"):
        reasons.append("heavy skew: linear models get a power transform")
    if bal:
        reasons.append("class imbalance: class weights inside every classifier")
    if signals.get("high_cardinality"):
        reasons.append(f"high-cardinality columns {signals['high_cardinality']}: encoded inside the folds")

    tree_cap = {"max_depth": 6, "min_samples_leaf": 5} if small else {}
    if fam == "classification":
        add("logreg", {"C": 0.3 if signals.get("high_vif") else 1.0}, "linear baseline with odds-ratio coefficients",
            favored=bool(signals.get("high_vif")))
        add("logreg_l1", {"C": 0.5, "l1_ratio": 0.5}, "regularized linear model (elastic net)", favored=bool(signals.get("high_vif")))
        add("random_forest", {"n_estimators": n_est, **tree_cap}, "robust non-linear default for tabular data")
        add("gradient_boosting", {"max_iter": 150 if fast else 250, **({"max_depth": 3} if small else {})},
            "histogram gradient boosting, handles missing values natively")
        add("xgboost", {"n_estimators": n_est, "max_depth": 3 if small else 4}, "gradient-boosted trees (XGBoost)")
        add("lightgbm", {"n_estimators": n_est, "num_leaves": 15 if small else 31, "min_child_samples": 10 if small else 20},
            "leaf-wise gradient boosting (LightGBM)")
        if plan.get("explainability_required"):
            add("decision_tree", {"max_depth": 4}, "shallow tree the client can read, because explainability was requested")
    elif fam == "regression":
        add("ridge", {"alpha": 3.0 if signals.get("high_vif") else 1.0}, "L2-regularized linear model",
            favored=bool(signals.get("high_vif")))
        add("lasso", {"alpha": 0.01}, "L1-regularized linear model with feature selection", favored=bool(signals.get("high_vif")))
        add("random_forest", {"n_estimators": n_est, **tree_cap}, "robust non-linear default")
        add("gradient_boosting", {"max_iter": 150 if fast else 250, **({"max_depth": 3} if small else {})},
            "histogram gradient boosting")
        add("xgboost", {"n_estimators": n_est, "max_depth": 3 if small else 4}, "gradient-boosted trees (XGBoost)")
        if plan.get("explainability_required"):
            add("decision_tree", {"max_depth": 4}, "shallow readable tree, because explainability was requested")
    elif fam == "clustering":
        cc = cluster_choice or {}
        k = int(cc.get("k", 3))
        add("kmeans", {"k": k}, f"k = {k} chosen by silhouette over k = 2..8")
        add("gmm", {"k": int(cc.get("k_gmm", k))}, "Gaussian mixture: allows elliptical, overlapping clusters")
        if n_features <= 15 and n_rows <= 50000:
            add("dbscan", {"eps": float(cc.get("eps", 0.5)), "min_samples": int(cc.get("min_samples", 5))},
                "density-based: finds irregular shapes and leaves noise unassigned")
        else:
            reasons.append("DBSCAN skipped: too many dimensions or rows for density estimates")
        if n_rows <= 5000:
            add("agglomerative", {"k": k}, "hierarchical (Ward) clustering")
        else:
            reasons.append("agglomerative clustering skipped: more than 5,000 rows")
    elif fam == "anomaly_detection":
        add("isolation_forest", {"n_estimators": 200}, "isolation forest: scales well, no distance assumptions")
        if n_rows <= 50000:
            add("lof", {"n_neighbors": 20}, "local outlier factor: density relative to neighbours")
        if n_rows <= 10000:
            add("ocsvm", {"nu": 0.05}, "one-class SVM: smooth boundary around normal points")
    elif fam == "time_series":
        add("seasonal_naive", {}, "the baseline any forecaster must beat")
        if n_rows >= 24:
            add("arima", {"order": [1, 1, 1]}, "seasonal ARIMA")
        add("gbm_lags", {}, "gradient boosting on lagged differences, recursive multi-step forecast")
    elif fam == "offline_rl":
        add("greedy_ridge", {"alpha": 1.0}, "direct-method policy from a linear reward model, valued by IPW")
        add("greedy_rf", {"n_estimators": n_est, "min_samples_leaf": 5}, "policy from a random-forest reward model")
        add("greedy_gbm", {"max_iter": 150}, "policy from a gradient-boosting reward model")
    kept = [s for s in specs if s["model"] not in exclude]
    if len(kept) >= 2 and len(kept) < len(specs):
        reasons.append("not fitted on this host (EXCLUDE_MODELS): " + ", ".join(sorted({s["model"] for s in specs} - {s["model"] for s in kept})))
        specs = kept
    for i, s in enumerate(specs):
        s["id"] = f"c{i + 1}_{s['model']}"
        s["name"] = s["model"].replace("_", " ")
    return jsonable({"specs": specs, "conditioning": reasons, "n_rows": n_rows, "n_features": n_features})


def choose_clusters(df: pd.DataFrame, task: dict, seed: int, k_max: int = 8) -> dict:
    """k by silhouette (KMeans), components by BIC (GMM), eps by the knee of the 5-NN distance curve."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors
    X, _ = pc.unsupervised_matrix(df, task, seed)
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), min(len(X), 3000), replace=False)]
    sil, bic = {}, {}
    for k in range(2, min(k_max, len(Xs) - 1) + 1):
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xs)
        sil[k] = float(silhouette_score(Xs, lab))
        bic[k] = float(GaussianMixture(n_components=k, random_state=seed).fit(Xs).bic(Xs))
    k_best = max(sil, key=sil.get)
    nn = NearestNeighbors(n_neighbors=5).fit(Xs)
    d = np.sort(nn.kneighbors(Xs)[0][:, -1])
    x = np.linspace(0, 1, len(d))
    y = (d - d.min()) / (d.max() - d.min() + 1e-12)
    knee = int(np.argmax(x - y))
    return jsonable({"k": k_best, "silhouette_by_k": sil, "k_gmm": min(bic, key=bic.get), "bic_by_k": bic,
                     "eps": float(d[knee]) if d[knee] > 0 else float(np.median(d)), "min_samples": 5,
                     "n_dims": int(X.shape[1])})


def evaluate_pipeline(spec: dict, df_train: pd.DataFrame, task: dict, seed: int, folds: int, repeats: int,
                      baseline_mean: float | None = None) -> dict:
    """Repeated k-fold CV (time-ordered or grouped when needed) on the training split."""
    res = pc.cv_evaluate(spec, df_train, task, seed, folds=folds, repeats=repeats)
    res["spec"] = {k: spec[k] for k in ("model", "params", "prep", "balance") if k in spec}
    if baseline_mean is not None and res["primary_mean"] == res["primary_mean"]:
        res["baseline_mean"] = baseline_mean
        res["gain_over_baseline"] = pc.oriented(task["metric"], res["primary_mean"]) - pc.oriented(task["metric"], baseline_mean)
    return jsonable(res)


def evaluate_baseline(df_train: pd.DataFrame, task: dict, seed: int, folds: int, repeats: int) -> dict:
    spec = pc.baseline_spec(task)
    res = pc.cv_evaluate(spec, df_train, task, seed, folds=folds, repeats=repeats)
    res["spec"] = spec
    return jsonable(res)


# --------------------------------------------------------------------------- tuning
SPACES: dict[str, dict[str, tuple]] = {
    "logreg": {"C": ("log", 0.01, 10.0)},
    "logreg_l1": {"C": ("log", 0.01, 10.0), "l1_ratio": ("float", 0.0, 1.0)},
    "ridge": {"alpha": ("log", 0.01, 100.0)},
    "lasso": {"alpha": ("log", 1e-4, 1.0)},
    "random_forest": {"max_depth": ("int", 3, 16), "min_samples_leaf": ("int", 1, 20), "max_features": ("float", 0.2, 1.0)},
    "gradient_boosting": {"learning_rate": ("log", 0.02, 0.3), "max_depth": ("int", 2, 8), "min_samples_leaf": ("int", 5, 60),
                          "l2_regularization": ("log", 1e-4, 5.0)},
    "xgboost": {"learning_rate": ("log", 0.02, 0.3), "max_depth": ("int", 2, 8), "min_child_weight": ("int", 1, 10),
                "subsample": ("float", 0.6, 1.0), "colsample_bytree": ("float", 0.5, 1.0), "reg_lambda": ("log", 0.1, 10.0)},
    "lightgbm": {"learning_rate": ("log", 0.02, 0.3), "num_leaves": ("int", 7, 63), "min_child_samples": ("int", 5, 60),
                 "subsample": ("float", 0.6, 1.0), "colsample_bytree": ("float", 0.5, 1.0)},
    "decision_tree": {"max_depth": ("int", 2, 8), "min_samples_leaf": ("int", 2, 40)},
    "kmeans": {}, "gmm": {"covariance_type": ("cat", "full", "diag", "tied")},
    "isolation_forest": {"n_estimators": ("int", 100, 400)}, "lof": {"n_neighbors": ("int", 5, 50)},
    "ocsvm": {"nu": ("float", 0.01, 0.2)}, "gbm_lags": {"lags": ("int", 2, 24), "max_iter": ("int", 50, 300)},
    "greedy_ridge": {"alpha": ("log", 0.01, 100.0)}, "greedy_rf": {"min_samples_leaf": ("int", 1, 30)},
    "greedy_gbm": {"max_iter": ("int", 50, 300)},
}


def tune_pipeline(spec: dict, df_train: pd.DataFrame, task: dict, seed: int, n_trials: int = 20,
                  timeout: float = 90.0, folds: int = 3) -> dict:
    """Optuna (TPE, seeded) on 3-fold CV of the primary metric. Only finalists are tuned."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = SPACES.get(spec["model"], {})
    if not space:
        return jsonable({"tuned": False, "reason": f"no search space for {spec['model']}", "best_params": spec.get("params", {})})
    metric = task["metric"]

    def objective(trial):
        params = dict(spec.get("params", {}))
        for name, (kind, *args) in space.items():
            if kind == "log":
                params[name] = trial.suggest_float(name, args[0], args[1], log=True)
            elif kind == "float":
                params[name] = trial.suggest_float(name, args[0], args[1])
            elif kind == "int":
                params[name] = trial.suggest_int(name, args[0], args[1])
            elif kind == "cat":
                params[name] = trial.suggest_categorical(name, list(args))
        res = pc.cv_evaluate({**spec, "params": params}, df_train, task, seed, folds=folds, repeats=1)
        v = res["primary_mean"]
        return pc.oriented(metric, v) if v == v else -1e9

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({k: v for k, v in (spec.get("params") or {}).items() if k in space})
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = {**spec.get("params", {}), **study.best_params}
    default_value = study.trials[0].value if study.trials else None
    return jsonable({"tuned": True, "best_params": best, "best_value_oriented": study.best_value,
                     "default_value_oriented": default_value, "n_trials": len(study.trials), "metric": metric})


def bootstrap_better(a: list[float], b: list[float], metric: str, seed: int, n: int = 2000) -> dict:
    """P(candidate a beats b) from paired per-fold differences (unpaired if fold counts differ)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    sign = 1 if pc.greater_is_better(metric) else -1
    if len(a) == len(b) and len(a) > 1:
        d = sign * (a - b)
        means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
        paired = True
    else:
        ma = a[rng.integers(0, len(a), (n, len(a)))].mean(axis=1)
        mb = b[rng.integers(0, len(b), (n, len(b)))].mean(axis=1)
        means = sign * (ma - mb)
        paired = False
    lo, hi = np.percentile(means, [2.5, 97.5])
    return jsonable({"p_better": float((means > 0).mean() + 0.5 * (means == 0).mean()), "mean_diff": float(means.mean()),
                     "ci95": [float(lo), float(hi)], "paired": paired, "n_boot": n})


def holdout_gain(chosen: dict, baseline: dict, metric: str, seed: int, n_boot: int = 2000) -> dict:
    """Does the chosen model beat the baseline on the held-out rows by more than chance? Paired bootstrap
    over those rows: both models are scored on the same resample, and the gain counts only when its 95%
    interval lies above zero. `chosen` and `baseline` are fit_and_score predictions (classification or
    regression). The gain is oriented so that positive is better, also for error metrics."""
    y = np.asarray(chosen["y_true"])
    n = len(y)
    if n < 20:
        return {"applicable": False, "reason": f"only {n} held-out rows"}
    sign = 1 if pc.greater_is_better(metric) else -1
    classes, pos = np.asarray(chosen.get("classes") or []), chosen.get("pos")
    preds = [(np.asarray(p["y_pred"]), np.asarray(p["proba"]) if p.get("proba") is not None else None)
             for p in (chosen, baseline)]

    def gain(idx) -> float:
        a, b = (pc.supervised_scores([metric], y[idx], yp[idx], pr[idx] if pr is not None else None, classes, pos)[metric]
                for yp, pr in preds)
        return sign * (a - b)

    point = gain(np.arange(n))
    rng = np.random.default_rng(seed)
    diffs = np.array([gain(rng.integers(0, n, n)) for _ in range(n_boot)])
    diffs = diffs[np.isfinite(diffs)]   # a resample can miss a class, leaving the metric undefined
    if len(diffs) < n_boot // 2:
        return {"applicable": False, "reason": f"{metric} undefined on most resamples of {n} rows"}
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return jsonable({"applicable": True, "metric": metric, "gain": point, "ci95": [lo, hi],
                     "p_better": float((diffs > 0).mean()), "meaningful": bool(lo > 0), "n_rows": n,
                     "n_boot": int(len(diffs)), "method": "paired bootstrap over the held-out rows"})


def bounded_metric(metric: str) -> bool:
    return metric in ("f1_macro", "roc_auc", "pr_auc", "recall_at_precision", "silhouette", "r2", "score_stability", "policy_value")


def evidence_confidence(chosen: dict, runner: dict | None, baseline_mean: float | None, metric: str, seed: int) -> dict:
    """The documented confidence formula (paper appendix):
        stability  s = exp(-sd / 0.05), with sd relative to |mean| for unbounded error metrics
        separation p = P(chosen beats runner-up), paired bootstrap over CV folds (0.5 if no runner-up)
        effect     e = 1 - exp(-d / 2), d = (chosen - baseline) / sd, oriented so higher is better
        confidence = 0.3 s + 0.4 p + 0.3 e"""
    mean, sd = chosen["primary_mean"], max(chosen["primary_std"], 1e-9)
    rel = sd if bounded_metric(metric) else sd / max(abs(mean), 1e-9)
    s = math.exp(-rel / 0.05)
    if runner is not None:
        bb = bootstrap_better(chosen["per_fold"], runner["per_fold"], metric, seed)
        p = bb["p_better"]
    else:
        bb, p = None, 0.5
    if baseline_mean is not None:
        d = (pc.oriented(metric, mean) - pc.oriented(metric, baseline_mean)) / sd
        e = 1 - math.exp(-max(0.0, d) / 2)
    else:
        d, e = None, 0.5
    conf = 0.3 * s + 0.4 * p + 0.3 * e
    return jsonable({"confidence": min(1.0, max(0.0, conf)), "stability": s, "separation": p, "effect": e,
                     "effect_d": d, "bootstrap": bb, "formula": "0.3*stability + 0.4*separation + 0.3*effect"})
