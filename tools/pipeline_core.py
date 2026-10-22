"""DS Cortex pipeline core: cleaning, holdout split, preprocessing, models, metrics,
cross-validation and final fit-and-score for every task family.

This file is deliberately self-contained (numpy, pandas, scikit-learn, and optionally
xgboost, lightgbm, statsmodels). export_pipeline_code() inlines it into the exported
script, so the script runs exactly the code that produced the reported score.
All fitting (imputers, encoders, scalers) happens inside the sklearn Pipeline, so inside
each cross-validation fold."""
from __future__ import annotations

import math
import os
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Threads per model: N_JOBS, else the CPUs joblib finds (it reads the container's CPU quota; -1 in
# LightGBM and XGBoost means one thread per host core, which stalls a container limited to a fraction of one)
N_JOBS = int(os.getenv("N_JOBS") or 0) or __import__("joblib").cpu_count()

LOWER_IS_BETTER = {"rmse", "mae", "smape", "davies_bouldin"}
TREE_MODELS = {"random_forest", "gradient_boosting", "xgboost", "lightgbm", "decision_tree", "isolation_forest",
               "greedy_rf", "greedy_gbm", "gbm_lags"}
INTERPRETABLE = {"logreg", "logreg_l1", "ridge", "lasso", "decision_tree", "kmeans", "seasonal_naive", "arima",
                 "greedy_ridge"}


def greater_is_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


def oriented(metric: str, value: float) -> float:
    return value if greater_is_better(metric) else -value


# =========================================================================== cleaning
def apply_cleaning_plan(df: pd.DataFrame, plan: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Apply approved, stateless cleaning actions (safe before splitting). Returns the new frame
    and a change log. Stateful fixes (imputation, capping) live in the pipeline instead."""
    out = df.copy()
    log: list[dict] = []
    for a in plan:
        if not a.get("enabled", True) or a.get("stage", "cleaning") != "cleaning":
            continue
        act, col, p = a["action"], a.get("column"), a.get("params") or {}
        before = len(out)
        if col is not None and col not in out.columns and act != "drop_duplicates":
            log.append({"action": act, "column": col, "detail": "column not present; skipped"})
            continue
        if act == "drop_column":
            out = out.drop(columns=[col])
            detail = "dropped"
        elif act == "drop_duplicates":
            out = out.drop_duplicates(keep="first")
            detail = f"removed {before - len(out)} exact duplicate rows"
        elif act == "drop_rows_missing_target":
            out = out[out[col].notna()]
            detail = f"removed {before - len(out)} rows with a missing target"
        elif act == "recode_case":
            mask = out[col].notna()
            out.loc[mask, col] = out.loc[mask, col].astype(str).str.strip().str.lower()
            detail = "lower-cased and stripped categories"
        elif act == "strip_whitespace":
            mask = out[col].notna()
            out.loc[mask, col] = out.loc[mask, col].astype(str).str.strip()
            detail = "stripped whitespace"
        elif act == "to_numeric":
            out[col] = pd.to_numeric(out[col].astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce")
            detail = "converted text to numbers"
        elif act == "to_datetime":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
            detail = "parsed dates"
        elif act == "sample_rows":
            n = int(p["n"])
            if len(out) > n:
                if p.get("time_column") and p["time_column"] in out:      # time series: keep the most recent rows
                    out = out.sort_values(p["time_column"]).tail(n)
                else:
                    strat = out[p["stratify"]] if p.get("stratify") in out and out[p["stratify"]].value_counts().min() >= 2 else None
                    out, _ = train_test_split(out, train_size=n, random_state=int(p.get("seed", 0)), stratify=strat)
            detail = f"sampled {len(out)} of {before} rows (row cap for this instance)"
        elif act == "set_nan_invalid":
            lo, hi = p.get("min"), p.get("max")
            bad = pd.Series(False, index=out.index)
            if lo is not None:
                bad |= out[col] < lo
            if hi is not None:
                bad |= out[col] > hi
            out.loc[bad, col] = np.nan
            detail = f"set {int(bad.sum())} impossible values to missing"
        else:
            detail = "unknown action; skipped"
        log.append({"action": act, "column": col, "rows_before": before, "rows_after": len(out), "detail": detail})
    return out.reset_index(drop=True), log


# =========================================================================== split
def holdout_split(df: pd.DataFrame, task: dict, seed: int, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a final test set once, at the start. Time series: the last 20 percent by time."""
    fam, target = task["task_family"], task.get("target")
    idx = np.arange(len(df))
    if fam == "clustering":
        return idx, idx
    if fam == "time_series":
        order = np.argsort(pd.to_datetime(df[task["time_column"]]).values, kind="stable")
        n_test = max(1, int(round(len(df) * test_size)))
        return np.sort(order[:-n_test]), np.sort(order[-n_test:])
    if task.get("group_column") and task["group_column"] in df:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(df, groups=df[task["group_column"]]))
        return tr, te
    strat = None
    if fam in ("classification", "anomaly_detection") and target:
        counts = df[target].value_counts()
        if counts.min() >= 2:
            strat = df[target]
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
    return np.sort(tr), np.sort(te)


# =========================================================================== features
def feature_columns(df: pd.DataFrame, task: dict, max_onehot: int = 30) -> dict[str, list[str]]:
    exclude = {task.get("target"), task.get("time_column") if task["task_family"] == "time_series" else None,
               task.get("action_column"), task.get("reward_column"), task.get("propensity_column"),
               task.get("group_column")}
    num, cat, high, dt, dropped = [], [], [], [], []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            num.append(c)
        elif pd.api.types.is_datetime64_any_dtype(s):
            dt.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            num.append(c)
        else:
            nun = s.nunique(dropna=True)
            ratio = nun / max(1, s.notna().sum())
            if nun <= 1:
                dropped.append(c)
            elif ratio > 0.9 and nun > 50:
                dropped.append(c)          # id-like or free text: not usable as a feature
            elif nun > max_onehot:
                high.append(c)
            else:
                cat.append(c)
    return {"numeric": num, "categorical": cat, "high_cardinality": high, "datetime": dt, "dropped": dropped}


class DateFeatures(BaseEstimator, TransformerMixin):
    """Year, month, day of week and day of year from datetime columns (stateless)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        parts = []
        for c in X.columns:
            d = pd.to_datetime(X[c], errors="coerce")
            parts += [d.dt.year, d.dt.month, d.dt.dayofweek, d.dt.dayofyear]
        return np.column_stack([p.astype(float).values for p in parts]) if parts else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        cols = list(input_features) if input_features is not None else []
        return np.array([f"{c}_{p}" for c in cols for p in ("year", "month", "dow", "doy")])


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap values at IQR fences learned on the training fold only."""

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lo_, self.hi_ = q1 - self.k * iqr, q3 + self.k * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by its frequency in the training fold."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X).astype(str)
        self.maps_ = [X[c].value_counts(normalize=True).to_dict() for c in X.columns]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).astype(str)
        return np.column_stack([X[c].map(m).fillna(0.0).values for c, m in zip(X.columns, self.maps_)])

    def get_feature_names_out(self, input_features=None):
        cols = list(input_features) if input_features is not None else []
        return np.array([f"{c}_freq" for c in cols])


def build_preprocessor(prep: dict, cols: dict[str, list[str]], supervised: bool, seed: int) -> ColumnTransformer:
    num_steps: list = [("impute", SimpleImputer(strategy=prep.get("impute_num", "median"),
                                                 add_indicator=bool(prep.get("add_indicator", False))))]
    if prep.get("winsorize"):
        num_steps.append(("winsorize", Winsorizer()))
    if prep.get("power"):
        num_steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))
    if prep.get("scale"):
        num_steps.append(("scale", StandardScaler()))
    transformers = []
    if cols["numeric"]:
        transformers.append(("num", Pipeline(num_steps), cols["numeric"]))
    if cols["categorical"]:
        cat_steps = [("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5))]
        if prep.get("scale"):
            cat_steps.append(("scale", StandardScaler()))
        transformers.append(("cat", Pipeline(cat_steps), cols["categorical"]))
    if cols["high_cardinality"]:
        enc = prep.get("high_card_encoding", "target" if supervised else "frequency")
        if enc == "target" and supervised:
            from sklearn.preprocessing import TargetEncoder
            hc = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                           ("target_enc", TargetEncoder(random_state=seed))])
        else:
            hc = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                           ("freq_enc", FrequencyEncoder())])
        transformers.append(("high_card", hc, cols["high_cardinality"]))
    if cols["datetime"]:
        date_steps = [("parts", DateFeatures()), ("impute", SimpleImputer(strategy="median"))]
        if prep.get("scale"):   # an unscaled year (about 2000) shrinks saga's step size until it barely moves
            date_steps.append(("scale", StandardScaler()))
        transformers.append(("date", Pipeline(date_steps), cols["datetime"]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def describe_steps(spec: dict, cols: dict[str, list[str]]) -> list[str]:
    p = spec.get("prep", {})
    steps = [f"impute numeric ({p.get('impute_num', 'median')})" + (" + missing indicators" if p.get("add_indicator") else "")]
    if p.get("winsorize"):
        steps.append("cap outliers at IQR fences (fitted per fold)")
    if p.get("power"):
        steps.append("Yeo-Johnson power transform")
    if p.get("scale"):
        steps.append("standard scaling")
    if cols.get("categorical"):
        steps.append(f"one-hot encode {len(cols['categorical'])} categorical column(s)")
    if cols.get("high_cardinality"):
        steps.append(f"{p.get('high_card_encoding', 'target')} encode {', '.join(cols['high_cardinality'])}")
    if cols.get("datetime"):
        steps.append("date parts from " + ", ".join(cols["datetime"]))
    if spec.get("balance"):
        steps.append(f"class balancing ({spec['balance']})")
    steps.append(f"model: {spec['model']}" + (f" {spec['params']}" if spec.get("params") else ""))
    return steps


# =========================================================================== models
def make_model(spec: dict, task: dict, seed: int, n_classes: int = 2, pos_weight: float = 1.0):
    m, p = spec["model"], dict(spec.get("params") or {})
    fam = task["task_family"]
    bal = spec.get("balance") == "class_weight"
    if fam == "classification":
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        cw = "balanced" if bal else None
        if m == "dummy":
            return DummyClassifier(strategy="prior")
        if m == "dummy_stratified":
            return DummyClassifier(strategy="stratified", random_state=seed)
        if m == "logreg":
            return LogisticRegression(C=p.get("C", 1.0), max_iter=3000, class_weight=cw)
        if m == "logreg_l1":
            return LogisticRegression(C=p.get("C", 0.5), l1_ratio=p.get("l1_ratio", 0.5), solver="saga",
                                      max_iter=5000, class_weight=cw, random_state=seed)
        if m == "decision_tree":
            return DecisionTreeClassifier(max_depth=p.get("max_depth", 4), min_samples_leaf=p.get("min_samples_leaf", 10),
                                          class_weight=cw, random_state=seed)
        if m == "random_forest":
            return RandomForestClassifier(n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth"),
                                          min_samples_leaf=p.get("min_samples_leaf", 1), max_features=p.get("max_features", "sqrt"),
                                          class_weight=cw, n_jobs=N_JOBS, random_state=seed)
        if m == "gradient_boosting":
            return HistGradientBoostingClassifier(learning_rate=p.get("learning_rate", 0.08), max_depth=p.get("max_depth"),
                                                  max_iter=p.get("max_iter", 200), min_samples_leaf=p.get("min_samples_leaf", 20),
                                                  l2_regularization=p.get("l2_regularization", 0.0), class_weight=cw,
                                                  random_state=seed)
        if m == "xgboost":
            from xgboost import XGBClassifier
            kw = {"scale_pos_weight": pos_weight} if bal and n_classes == 2 else {}
            return XGBClassifier(n_estimators=p.get("n_estimators", 300), max_depth=p.get("max_depth", 4),
                                 learning_rate=p.get("learning_rate", 0.08), subsample=p.get("subsample", 0.9),
                                 colsample_bytree=p.get("colsample_bytree", 0.9), min_child_weight=p.get("min_child_weight", 1),
                                 reg_lambda=p.get("reg_lambda", 1.0), tree_method="hist", n_jobs=N_JOBS, random_state=seed,
                                 eval_metric="logloss", verbosity=0, **kw)
        if m == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=p.get("n_estimators", 300), learning_rate=p.get("learning_rate", 0.05),
                                  num_leaves=p.get("num_leaves", 31), min_child_samples=p.get("min_child_samples", 20),
                                  subsample=p.get("subsample", 0.9), subsample_freq=1, colsample_bytree=p.get("colsample_bytree", 0.9),
                                  reg_lambda=p.get("reg_lambda", 0.0), class_weight=cw, random_state=seed, n_jobs=N_JOBS, verbose=-1)
    if fam in ("regression", "offline_rl"):
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Lasso, Ridge
        from sklearn.tree import DecisionTreeRegressor
        m = {"greedy_ridge": "ridge", "greedy_rf": "random_forest", "greedy_gbm": "gradient_boosting"}.get(m, m)
        if m == "dummy":
            return DummyRegressor(strategy="mean")
        if m in ("ridge", "lasso"):
            # standardize the target so the penalty is scale-free (and Lasso converges on large targets)
            from sklearn.compose import TransformedTargetRegressor
            base = Ridge(alpha=p.get("alpha", 1.0)) if m == "ridge" else Lasso(alpha=p.get("alpha", 0.01), max_iter=20000)
            return base if fam == "offline_rl" else TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
        if m == "decision_tree":
            return DecisionTreeRegressor(max_depth=p.get("max_depth", 5), min_samples_leaf=p.get("min_samples_leaf", 10),
                                         random_state=seed)
        if m == "random_forest":
            return RandomForestRegressor(n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth"),
                                         min_samples_leaf=p.get("min_samples_leaf", 1), max_features=p.get("max_features", 1.0),
                                         n_jobs=N_JOBS, random_state=seed)
        if m == "gradient_boosting":
            return HistGradientBoostingRegressor(learning_rate=p.get("learning_rate", 0.08), max_depth=p.get("max_depth"),
                                                 max_iter=p.get("max_iter", 200), min_samples_leaf=p.get("min_samples_leaf", 20),
                                                 l2_regularization=p.get("l2_regularization", 0.0), random_state=seed)
        if m == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=p.get("n_estimators", 300), max_depth=p.get("max_depth", 4),
                                learning_rate=p.get("learning_rate", 0.08), subsample=p.get("subsample", 0.9),
                                colsample_bytree=p.get("colsample_bytree", 0.9), min_child_weight=p.get("min_child_weight", 1),
                                reg_lambda=p.get("reg_lambda", 1.0), tree_method="hist", n_jobs=N_JOBS, random_state=seed, verbosity=0)
        if m == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=p.get("n_estimators", 300), learning_rate=p.get("learning_rate", 0.05),
                                 num_leaves=p.get("num_leaves", 31), min_child_samples=p.get("min_child_samples", 20),
                                 subsample=p.get("subsample", 0.9), subsample_freq=1, colsample_bytree=p.get("colsample_bytree", 0.9),
                                 reg_lambda=p.get("reg_lambda", 0.0), random_state=seed, n_jobs=N_JOBS, verbose=-1)
    if fam == "clustering":
        from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
        from sklearn.mixture import GaussianMixture
        if m == "kmeans":
            return KMeans(n_clusters=p.get("k", 3), n_init=10, random_state=seed)
        if m == "gmm":
            return GaussianMixture(n_components=p.get("k", 3), covariance_type=p.get("covariance_type", "full"), random_state=seed)
        if m == "dbscan":
            return DBSCAN(eps=p.get("eps", 0.5), min_samples=p.get("min_samples", 5))
        if m == "agglomerative":
            return AgglomerativeClustering(n_clusters=p.get("k", 3), linkage=p.get("linkage", "ward"))
        if m == "random_labels":
            return RandomLabels(k=p.get("k", 3), seed=seed)
    if fam == "anomaly_detection":
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.svm import OneClassSVM
        if m == "isolation_forest":
            return IsolationForest(n_estimators=p.get("n_estimators", 200), contamination="auto", random_state=seed, n_jobs=N_JOBS)
        if m == "lof":
            return LocalOutlierFactor(n_neighbors=p.get("n_neighbors", 20), novelty=True)
        if m == "ocsvm":
            return OneClassSVM(nu=p.get("nu", 0.05), gamma="scale")
        if m == "random_scores":
            return RandomScores(seed=seed)
    raise ValueError(f"Unknown model '{m}' for task family '{fam}'")


class RandomLabels(BaseEstimator):
    """Clustering baseline: random assignment to k groups."""

    def __init__(self, k: int = 3, seed: int = 0):
        self.k, self.seed = k, seed

    def fit_predict(self, X, y=None):
        return np.random.default_rng(self.seed).integers(0, self.k, len(X))


class RandomScores(BaseEstimator):
    """Anomaly baseline: random scores."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y=None):
        return self

    def score_samples(self, X):
        return np.random.default_rng(self.seed + len(X)).random(len(X))


def make_pipeline(spec: dict, task: dict, cols: dict, seed: int, n_classes: int = 2, pos_weight: float = 1.0) -> Pipeline:
    supervised = task["task_family"] in ("classification", "regression", "offline_rl")
    prep = build_preprocessor(spec.get("prep", {}), cols, supervised, seed)
    return Pipeline([("prep", prep), ("model", make_model(spec, task, seed, n_classes, pos_weight))])


# =========================================================================== metrics
def smape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(denom == 0, 0.0, 2 * np.abs(y_pred - y_true) / denom)
    return float(100 * np.mean(v))


def recall_at_precision(y_true_bin, scores, min_precision: float = 0.8) -> float:
    p, r, _ = precision_recall_curve(y_true_bin, scores)
    ok = r[p >= min_precision]
    return float(ok.max()) if len(ok) else 0.0


def supervised_scores(metrics: list[str], y_true, y_pred, proba, classes: np.ndarray, pos: int | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in metrics:
        try:
            if m == "f1_macro":
                out[m] = f1_score(y_true, y_pred, average="macro")
            elif m == "roc_auc":
                if len(classes) == 2:
                    out[m] = roc_auc_score(y_true, proba[:, 1])
                else:
                    out[m] = roc_auc_score(y_true, proba, multi_class="ovr", labels=np.arange(len(classes)))
            elif m == "pr_auc":
                if len(classes) == 2:
                    out[m] = average_precision_score(y_true == pos, proba[:, pos])
                else:
                    onehot = np.eye(len(classes))[y_true]
                    out[m] = average_precision_score(onehot, proba, average="macro")
            elif m == "recall_at_precision":
                if len(classes) == 2:
                    out[m] = recall_at_precision(y_true == pos, proba[:, pos])
                else:   # one-vs-rest, averaged over classes like pr_auc above
                    out[m] = float(np.mean([recall_at_precision(y_true == j, proba[:, j]) for j in range(len(classes))]))
            elif m == "rmse":
                out[m] = math.sqrt(mean_squared_error(y_true, y_pred))
            elif m == "mae":
                out[m] = mean_absolute_error(y_true, y_pred)
            elif m == "r2":
                out[m] = r2_score(y_true, y_pred)
        except ValueError:
            out[m] = float("nan")
    return {k: float(v) for k, v in out.items()}


# =========================================================================== data preparation per family
def prepare_supervised(df: pd.DataFrame, task: dict, max_onehot: int = 30) -> dict:
    target = task["target"]
    d = df[df[target].notna()].reset_index(drop=True)
    cols = feature_columns(d, task, max_onehot)
    X = d[cols["numeric"] + cols["categorical"] + cols["high_cardinality"] + cols["datetime"]]
    info: dict[str, Any] = {"X": X, "cols": cols, "groups": d[task["group_column"]].values if task.get("group_column") in d else None}
    if task["task_family"] == "classification":
        classes, y = np.unique(d[target].astype(str), return_inverse=True)
        counts = np.bincount(y, minlength=len(classes))
        info.update(y=y, classes=classes, pos=int(np.argmin(counts)) if len(classes) == 2 else None,
                    pos_weight=float(counts.max() / max(1, counts.min())))
    else:
        info.update(y=pd.to_numeric(d[target], errors="coerce").values.astype(float), classes=np.array([]), pos=None, pos_weight=1.0)
    return info


def _fit_predict_supervised(spec, task, info, tr, te, seed, metrics):
    pipe = make_pipeline(spec, task, info["cols"], seed, len(info["classes"]) or 2, info["pos_weight"])
    X, y = info["X"], info["y"]
    t0 = time.perf_counter()
    pipe.fit(X.iloc[tr], y[tr])
    fit_time = time.perf_counter() - t0
    out = {}
    for name, idx in (("val", te), ("train", tr)):
        pred = pipe.predict(X.iloc[idx])
        proba = pipe.predict_proba(X.iloc[idx]) if task["task_family"] == "classification" else None
        if proba is not None and proba.shape[1] < len(info["classes"]):   # class absent from a fold
            full = np.zeros((len(idx), len(info["classes"])))
            full[:, pipe.classes_] = proba
            proba = full
        out[name] = supervised_scores(metrics, y[idx], pred, proba, info["classes"], info["pos"])
    return out, fit_time, pipe


def splitter(task: dict, n: int, folds: int, repeats: int, seed: int, y=None, groups=None):
    fam = task["task_family"]
    if groups is not None:
        return GroupKFold(n_splits=min(folds, len(np.unique(groups)))).split(np.zeros(n), y, groups)
    if fam == "classification":
        min_count = int(np.bincount(y).min())
        k = max(2, min(folds, min_count))
        return RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed).split(np.zeros(n), y)
    return RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed).split(np.zeros(n))


# =========================================================================== time series
def infer_season(dates: pd.Series, y: np.ndarray) -> int:
    freq = None
    try:
        freq = pd.infer_freq(pd.DatetimeIndex(dates))
    except (TypeError, ValueError):
        pass
    table = {"M": 12, "MS": 12, "ME": 12, "W": 52, "D": 7, "B": 5, "H": 24, "h": 24, "Q": 4, "QS": 4, "QE": 4}
    if freq:
        base = freq.split("-")[0]
        for k, v in table.items():
            if base == k or base.rstrip("0123456789") == k:
                return v if len(y) >= 2 * v else 1
    n = len(y)
    if n < 8:
        return 1
    yc = y - y.mean()
    acf = [np.corrcoef(yc[:-k], yc[k:])[0, 1] for k in range(2, min(n // 2, 60))]
    if acf and np.nanmax(acf) > 0.3:
        return int(np.nanargmax(acf) + 2)
    return 1


def ts_forecast(spec: dict, y_train: np.ndarray, h: int, m: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (forecast, half-width of a 95 percent interval or None)."""
    model = spec["model"]
    y_train = np.asarray(y_train, float)
    if model in ("naive", "seasonal_naive"):
        season = m if model == "seasonal_naive" and m > 1 and len(y_train) >= m else 1
        last = y_train[-season:]
        fc = np.array([last[i % season] for i in range(h)])
        resid = y_train[season:] - y_train[:-season] if len(y_train) > season else np.array([0.0])
        return fc, 1.96 * np.std(resid) * np.sqrt(1 + np.arange(h) // max(1, season))
    if model == "arima":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        seasonal = (0, 1, 1, m) if m > 1 and len(y_train) >= 3 * m else (0, 0, 0, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(y_train, order=tuple(spec.get("params", {}).get("order", (1, 1, 1))), seasonal_order=seasonal,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            f = res.get_forecast(h)
            ci = np.asarray(f.conf_int(alpha=0.05))
            mean = np.asarray(f.predicted_mean)
        return mean, (ci[:, 1] - ci[:, 0]) / 2
    if model == "gbm_lags":
        from sklearn.ensemble import HistGradientBoostingRegressor
        L = int(spec.get("params", {}).get("lags", max(3, min(m if m > 1 else 6, 24))))
        d = np.diff(y_train)
        if len(d) <= L + 5:
            return ts_forecast({"model": "naive"}, y_train, h, m, seed)
        Xs, ys = [], []
        for t in range(L, len(d)):
            Xs.append(np.r_[d[t - L:t][::-1], (t + 1) % max(m, 1)])
            ys.append(d[t])
        reg = HistGradientBoostingRegressor(max_iter=spec.get("params", {}).get("max_iter", 200),
                                            learning_rate=0.05, random_state=seed).fit(np.array(Xs), np.array(ys))
        resid = np.array(ys) - reg.predict(np.array(Xs))
        hist, level, fc = list(d), y_train[-1], []
        for _ in range(h):
            t = len(hist)
            x = np.r_[np.array(hist[-L:])[::-1], (t + 1) % max(m, 1)]
            dd = float(reg.predict(x.reshape(1, -1))[0])
            hist.append(dd)
            level += dd
            fc.append(level)
        return np.array(fc), 1.96 * np.std(resid) * np.sqrt(np.arange(1, h + 1))
    raise ValueError(f"Unknown time-series model {model}")


def prepare_series(df: pd.DataFrame, task: dict) -> tuple[pd.Series, np.ndarray]:
    d = df[[task["time_column"], task["target"]]].copy()
    d[task["time_column"]] = pd.to_datetime(d[task["time_column"]], errors="coerce")
    d = d.dropna().sort_values(task["time_column"])
    y = pd.to_numeric(d[task["target"]], errors="coerce").interpolate().bfill().values.astype(float)
    return d[task["time_column"]].reset_index(drop=True), y


# =========================================================================== clustering / anomaly helpers
def unsupervised_matrix(df: pd.DataFrame, task: dict, seed: int, scale: bool = True) -> tuple[np.ndarray, dict]:
    cols = feature_columns(df, task)
    prep = build_preprocessor({"scale": scale}, cols, supervised=False, seed=seed)
    X = df[cols["numeric"] + cols["categorical"] + cols["high_cardinality"] + cols["datetime"]]
    return np.asarray(prep.fit_transform(X), float), cols


def cluster_scores(X: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    mask = labels >= 0
    noise = float(1 - mask.mean())
    lab = labels[mask]
    if len(np.unique(lab)) < 2 or mask.sum() < 3:
        return {"silhouette": -1.0, "davies_bouldin": float("nan"), "noise_share": noise, "n_clusters": int(len(np.unique(lab)))}
    Xm = X[mask]
    sample = min(len(Xm), 5000)
    sil = silhouette_score(Xm, lab, sample_size=sample if sample < len(Xm) else None, random_state=seed)
    if noise > 0.3:
        sil = sil * (1 - noise)   # penalise models that discard much of the data as noise
    return {"silhouette": float(sil), "davies_bouldin": float(davies_bouldin_score(Xm, lab)),
            "noise_share": noise, "n_clusters": int(len(np.unique(lab)))}


def anomaly_scores(model, X_fit: np.ndarray, X_score: np.ndarray) -> np.ndarray:
    model.fit(X_fit)
    return -np.asarray(model.score_samples(X_score), float)   # higher = more anomalous


# =========================================================================== offline RL helpers
def bandit_parts(df: pd.DataFrame, task: dict) -> dict:
    a_col, r_col, p_col = task["action_column"], task["reward_column"], task.get("propensity_column")
    d = df[df[a_col].notna() & df[r_col].notna()].reset_index(drop=True)
    cols = feature_columns(d, task)
    X = d[cols["numeric"] + cols["categorical"] + cols["high_cardinality"] + cols["datetime"]]
    actions = np.array(sorted(d[a_col].astype(str).unique()))
    a = np.searchsorted(actions, d[a_col].astype(str).values)
    r = pd.to_numeric(d[r_col], errors="coerce").values.astype(float)
    p = pd.to_numeric(d[p_col], errors="coerce").values.astype(float) if p_col and p_col in d else None
    return {"X": X, "a": a, "r": r, "p": p, "actions": actions, "cols": cols}


def _bandit_fit_value(spec, task, parts, tr, te, seed) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    X, a, r, p, K = parts["X"], parts["a"], parts["r"], parts["p"], len(parts["actions"])
    prep = build_preprocessor({"scale": True}, parts["cols"], supervised=False, seed=seed)
    Xt_tr = np.asarray(prep.fit_transform(X.iloc[tr]), float)
    Xt_te = np.asarray(prep.transform(X.iloc[te]), float)
    if p is None:   # estimate logging propensities on the training rows only
        lr = LogisticRegression(max_iter=2000).fit(Xt_tr, a[tr])
        proba = np.zeros((len(te), K))
        proba[:, lr.classes_] = lr.predict_proba(Xt_te)
        p_te = proba[np.arange(len(te)), a[te]]
    else:
        p_te = p[te]
    p_te = np.clip(p_te, 0.01, 1.0)
    if spec["model"] == "logged":
        return {"policy_value": float(np.mean(r[te])), "dr_value": float(np.mean(r[te])), "match_rate": 1.0}
    reg = make_model(spec, {"task_family": "offline_rl"}, seed)
    A_tr = np.eye(K)[a[tr]]
    reg.fit(np.hstack([Xt_tr, A_tr]), r[tr])
    q = np.column_stack([reg.predict(np.hstack([Xt_te, np.tile(np.eye(K)[k], (len(te), 1))])) for k in range(K)])
    pi = q.argmax(axis=1)
    match = (pi == a[te]).astype(float)
    ipw = float(np.mean(match * r[te] / p_te))
    dr = float(np.mean(q[np.arange(len(te)), pi] + match / p_te * (r[te] - q[np.arange(len(te)), a[te]])))
    return {"policy_value": ipw, "dr_value": dr, "match_rate": float(match.mean())}


# =========================================================================== cross-validation
def cv_evaluate(spec: dict, df_train: pd.DataFrame, task: dict, seed: int, folds: int = 5, repeats: int = 3,
                metrics: list[str] | None = None) -> dict:
    """Repeated k-fold (time-ordered for series) on the TRAINING split only."""
    fam = task["task_family"]
    metrics = metrics or [task["metric"]] + list(task.get("secondary_metrics", []))
    primary = task["metric"]
    folds_out, train_out, secondary, times = [], [], {m: [] for m in metrics if m != primary}, []

    if fam in ("classification", "regression"):
        info = prepare_supervised(df_train, task)
        for tr, te in splitter(task, len(info["y"]), folds, repeats, seed, info["y"], info["groups"]):
            res, ft, _ = _fit_predict_supervised(spec, task, info, tr, te, seed, metrics)
            folds_out.append(res["val"][primary])
            train_out.append(res["train"][primary])
            for m in secondary:
                secondary[m].append(res["val"].get(m, float("nan")))
            times.append(ft)
    elif fam == "time_series":
        dates, y = prepare_series(df_train, task)
        m = infer_season(dates, y)
        h = max(1, min(len(y) // (folds + 1), int(task.get("horizon") or round(len(y) * 0.2))))
        k = max(2, min(folds, (len(y) - max(2 * m, 8)) // h))
        for tr, te in TimeSeriesSplit(n_splits=k, test_size=h).split(y):
            t0 = time.perf_counter()
            fc, _ = ts_forecast(spec, y[tr], len(te), m, seed)
            times.append(time.perf_counter() - t0)
            vals = {"smape": smape(y[te], fc), "mae": mean_absolute_error(y[te], fc)}
            folds_out.append(vals[primary])
            for mm in secondary:
                secondary[mm].append(vals.get(mm, float("nan")))
    elif fam == "clustering":
        X, _ = unsupervised_matrix(df_train, task, seed)
        rng = np.random.default_rng(seed)
        for r in range(max(3, folds)):
            idx = rng.choice(len(X), size=max(3, int(0.8 * len(X))), replace=False)
            model = make_model(spec, task, seed + r)
            t0 = time.perf_counter()
            labels = model.fit_predict(X[idx])
            times.append(time.perf_counter() - t0)
            sc = cluster_scores(X[idx], np.asarray(labels), seed)
            folds_out.append(sc[primary])
            for mm in secondary:
                secondary[mm].append(sc.get(mm, float("nan")))
    elif fam == "anomaly_detection":
        X, _ = unsupervised_matrix(df_train, task, seed)
        labeled = bool(task.get("target"))
        y = None
        if labeled:
            yy = df_train[task["target"]].astype(str).values
            vals, counts = np.unique(yy, return_counts=True)
            y = (yy == vals[np.argmin(counts)]).astype(int)
        kf = (StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y) if labeled
              else KFold(folds, shuffle=True, random_state=seed).split(X))
        for i, (tr, te) in enumerate(kf):
            t0 = time.perf_counter()
            if labeled:
                s = anomaly_scores(make_model(spec, task, seed), X[tr], X[te])
                val = average_precision_score(y[te], s) if y[te].sum() > 0 else float("nan")
            else:
                rng = np.random.default_rng(seed + i)
                half = rng.permutation(tr)
                s1 = anomaly_scores(make_model(spec, task, seed), X[half[: len(half) // 2]], X[te])
                s2 = anomaly_scores(make_model(spec, task, seed + 1), X[half[len(half) // 2:]], X[te])
                val = float(pd.Series(s1).corr(pd.Series(s2), method="spearman"))
            times.append(time.perf_counter() - t0)
            folds_out.append(val)
    elif fam == "offline_rl":
        parts = bandit_parts(df_train, task)
        for tr, te in RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed).split(parts["a"]):
            t0 = time.perf_counter()
            vals = _bandit_fit_value(spec, task, parts, tr, te, seed)
            times.append(time.perf_counter() - t0)
            folds_out.append(vals[primary])
            for mm in secondary:
                secondary[mm].append(vals.get(mm, float("nan")))
    else:
        raise ValueError(fam)

    arr = np.asarray(folds_out, float)
    return {
        "primary": primary,
        "primary_mean": float(np.nanmean(arr)) if len(arr) else float("nan"),
        "primary_std": float(np.nanstd(arr)) if len(arr) else float("nan"),
        "per_fold": [float(v) for v in arr],
        "secondary": {m: float(np.nanmean(v)) for m, v in secondary.items() if len(v)},
        "train_mean": float(np.nanmean(train_out)) if train_out else None,
        "fit_time": float(np.sum(times)),
        "n_splits": int(len(arr)),
    }


# =========================================================================== final fit and score
def fit_and_score(df: pd.DataFrame, task: dict, spec: dict, seed: int,
                  split: tuple[np.ndarray, np.ndarray] | None = None, keep_model: bool = True) -> dict:
    """Fit on the training split and score once on the held-out test split."""
    fam = task["task_family"]
    tr_idx, te_idx = split if split is not None else holdout_split(df, task, seed)
    train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    metrics = [task["metric"]] + list(task.get("secondary_metrics", []))
    out: dict[str, Any] = {"n_train": int(len(train)), "n_test": int(len(test)), "metric": task["metric"]}
    if fam in ("classification", "regression"):
        info_tr = prepare_supervised(train, task)
        test_rows = test[test[task["target"]].notna()].reset_index(drop=True)
        info_te = {"X": test_rows[list(info_tr["X"].columns)]}
        if fam == "classification":   # map test labels onto the training label order
            info_te["y"] = np.searchsorted(info_tr["classes"], test_rows[task["target"]].astype(str).values)
        else:
            info_te["y"] = pd.to_numeric(test_rows[task["target"]], errors="coerce").values.astype(float)
        pipe = make_pipeline(spec, task, info_tr["cols"], seed, len(info_tr["classes"]) or 2, info_tr["pos_weight"])
        pipe.fit(info_tr["X"], info_tr["y"])
        pred = pipe.predict(info_te["X"])
        proba = pipe.predict_proba(info_te["X"]) if fam == "classification" else None
        out["test"] = supervised_scores(metrics, info_te["y"], pred, proba, info_tr["classes"], info_tr["pos"])
        out["predictions"] = {"y_true": info_te["y"].tolist(), "y_pred": np.asarray(pred).tolist(),
                              "proba": proba.tolist() if proba is not None else None,
                              "classes": info_tr["classes"].tolist(), "pos": info_tr["pos"]}
        if keep_model:
            out["model"] = pipe
            out["X_test"] = info_te["X"]
            out["X_train"] = info_tr["X"]
            out["y_train"] = info_tr["y"]
    elif fam == "time_series":
        dates, y = prepare_series(df, task)
        n_test = len(te_idx)
        y_tr, y_te = y[:-n_test], y[-n_test:]
        m = infer_season(dates[:-n_test], y_tr)
        fc, half = ts_forecast(spec, y_tr, n_test, m, seed)
        out["test"] = {"smape": smape(y_te, fc), "mae": float(mean_absolute_error(y_te, fc))}
        out["predictions"] = {"dates": [str(d.date()) for d in dates], "y": y.tolist(), "n_test": n_test,
                              "forecast": fc.tolist(), "half_width": None if half is None else np.asarray(half).tolist(),
                              "season": m}
    elif fam == "clustering":
        X, cols = unsupervised_matrix(df, task, seed)
        model = make_model(spec, task, seed)
        labels = np.asarray(model.fit_predict(X))
        out["test"] = cluster_scores(X, labels, seed)
        out["predictions"] = {"labels": labels.tolist()}
        if keep_model:
            out["model"] = model
            out["X_matrix"] = X
    elif fam == "anomaly_detection":
        X_all, cols = unsupervised_matrix(df, task, seed)
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        model = make_model(spec, task, seed)
        s = anomaly_scores(model, X_tr, X_te)
        if task.get("target"):
            yy = df[task["target"]].astype(str).values
            vals, counts = np.unique(yy, return_counts=True)
            y_te = (yy[te_idx] == vals[np.argmin(counts)]).astype(int)
            out["test"] = {"pr_auc": float(average_precision_score(y_te, s)) if y_te.sum() else float("nan")}
            out["predictions"] = {"scores": s.tolist(), "y_true": y_te.tolist()}
        else:   # stability: two models fitted on disjoint halves of the training rows must agree
            half = np.random.default_rng(seed).permutation(len(X_tr))
            s_a = anomaly_scores(make_model(spec, task, seed), X_tr[half[: len(half) // 2]], X_te)
            s_b = anomaly_scores(make_model(spec, task, seed + 1), X_tr[half[len(half) // 2:]], X_te)
            out["test"] = {"score_stability": float(pd.Series(s_a).corr(pd.Series(s_b), method="spearman"))}
            out["predictions"] = {"scores": s.tolist()}
    elif fam == "offline_rl":
        parts = bandit_parts(df, task)
        out["test"] = _bandit_fit_value(spec, task, parts, tr_idx[tr_idx < len(parts["a"])], te_idx[te_idx < len(parts["a"])], seed)
        out["predictions"] = {"actions": parts["actions"].tolist()}
    out["score"] = float(out["test"].get(task["metric"], float("nan")))
    return out


def baseline_spec(task: dict) -> dict:
    fam = task["task_family"]
    return {"classification": {"model": "dummy"}, "regression": {"model": "dummy"},
            "time_series": {"model": "naive"}, "clustering": {"model": "random_labels", "params": {"k": 3}},
            "anomaly_detection": {"model": "random_scores"}, "offline_rl": {"model": "logged"}}[fam] | {"prep": {}}
