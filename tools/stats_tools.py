"""Statistics tools ("can we actually say that?"). Effect sizes are always reported next to
p-values, never p-values alone."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as st

from tools.common import jsonable

warnings.filterwarnings("ignore", category=RuntimeWarning)


def numeric_features(df: pd.DataFrame, exclude: set[str] | None = None, max_cols: int = 60) -> list[str]:
    exclude = exclude or set()
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique(dropna=True) > 2]
    if len(cols) > max_cols:
        cols = list(df[cols].var().sort_values(ascending=False).index[:max_cols])
    return cols


def categorical_features(df: pd.DataFrame, exclude: set[str] | None = None, max_levels: int = 50) -> list[str]:
    exclude = exclude or set()
    return [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c]) and 1 < df[c].nunique() <= max_levels]


def normality_tests(df: pd.DataFrame, cols: list[str]) -> dict:
    """Shapiro-Wilk under 5,000 rows, D'Agostino otherwise, plus skew and excess kurtosis."""
    out = {}
    for c in cols:
        x = df[c].dropna().astype(float).values
        if len(x) < 8 or np.std(x) == 0:
            continue
        if len(x) < 5000:
            stat, p = st.shapiro(x)
            test = "shapiro"
        else:
            stat, p = st.normaltest(x)
            test = "dagostino"
        out[c] = {"test": test, "statistic": float(stat), "p_value": float(p), "normal": bool(p >= 0.05),
                  "skew": float(st.skew(x)), "kurtosis": float(st.kurtosis(x)), "n": int(len(x))}
    return jsonable({"columns": out})


def homoscedasticity_test(df: pd.DataFrame, target: str, task_family: str, cols: list[str]) -> dict:
    """Levene across classes (classification) or Breusch-Pagan on OLS residuals (regression)."""
    if not cols or target not in df:
        return {"test": None}
    d = df[cols + [target]].dropna(subset=[target])
    if task_family == "classification":
        out = {}
        groups = d[target].astype(str)
        for c in cols:
            samples = [d.loc[groups == g, c].dropna().values for g in groups.unique()]
            samples = [s for s in samples if len(s) >= 3]
            if len(samples) >= 2:
                stat, p = st.levene(*samples)
                out[c] = {"statistic": float(stat), "p_value": float(p), "equal_variance": bool(p >= 0.01)}
        return jsonable({"test": "levene", "columns": out})
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    X = d[cols].fillna(d[cols].median())
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    X = sm.add_constant(X, has_constant="add")
    y = pd.to_numeric(d[target], errors="coerce").values
    res = sm.OLS(y, X).fit()
    lm, lm_p, f, f_p = het_breuschpagan(res.resid, X)
    return jsonable({"test": "breusch_pagan", "statistic": float(lm), "p_value": float(lm_p),
                     "heteroscedastic": bool(lm_p < 0.01), "ols_r2": float(res.rsquared)})


def independence_check(df: pd.DataFrame, target: str, cols: list[str], time_col: str | None = None) -> dict:
    """Durbin-Watson on OLS residuals in row (or time) order. Near 2 means no autocorrelation."""
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson
    d = df.copy()
    if time_col and time_col in d:
        d = d.sort_values(time_col)
    y = pd.to_numeric(d[target], errors="coerce")
    ok = y.notna()
    if ok.sum() < 10:
        return {"durbin_watson": None}
    if cols:
        X = d.loc[ok, cols].fillna(d[cols].median())
        X = sm.add_constant((X - X.mean()) / X.std(ddof=0).replace(0, 1), has_constant="add")
        resid = sm.OLS(y[ok].values, X).fit().resid
    else:
        resid = y[ok].values - y[ok].mean()
    dw = float(durbin_watson(resid))
    return jsonable({"durbin_watson": dw, "ordered_by": time_col or "row order",
                     "autocorrelated": bool(dw < 1.5 or dw > 2.5)})


def vif_report(df: pd.DataFrame, cols: list[str]) -> dict:
    """Variance inflation factor per numeric feature; above 5 is a warning, above 10 severe."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    cols = [c for c in cols if df[c].nunique() > 1][:40]
    if len(cols) < 2:
        return {"columns": {}, "mean_vif": 1.0}
    X = df[cols].astype(float).fillna(df[cols].median())
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    X.insert(0, "const", 1.0)
    out = {}
    for i, c in enumerate(cols, start=1):
        with np.errstate(divide="ignore"):
            v = float(variance_inflation_factor(X.values, i))
        v = 1e6 if not np.isfinite(v) else min(v, 1e6)
        out[c] = {"vif": v, "level": "severe" if v > 10 else "warning" if v > 5 else "ok"}
    vals = [min(r["vif"], 50.0) for r in out.values()]
    return jsonable({"columns": out, "mean_vif": float(np.mean(vals))})


def effect_sizes(df: pd.DataFrame, target: str, task_family: str, num_cols: list[str], cat_cols: list[str]) -> dict:
    """Cohen's d or eta-squared for numeric features, Cramer's V for categorical ones, with p-values."""
    d = df[df[target].notna()]
    out = {}
    if task_family == "classification":
        y = d[target].astype(str)
        classes = y.unique()
        for c in num_cols:
            groups = [d.loc[y == g, c].dropna().astype(float).values for g in classes]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) < 2:
                continue
            if len(groups) == 2:
                a, b = groups
                pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)) or 1.0
                eff = float(abs(a.mean() - b.mean()) / pooled)
                p = float(st.ttest_ind(a, b, equal_var=False).pvalue)
                out[c] = {"measure": "cohens_d", "effect": eff, "p_value": p}
            else:
                f, p = st.f_oneway(*groups)
                allv = np.concatenate(groups)
                ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
                ss_t = ((allv - allv.mean()) ** 2).sum() or 1.0
                out[c] = {"measure": "eta_squared", "effect": float(ss_b / ss_t), "p_value": float(p)}
        for c in cat_cols:
            tab = pd.crosstab(d[c].astype(str), y)
            if tab.shape[0] > 1 and tab.shape[1] > 1:
                chi2, p, _, _ = st.chi2_contingency(tab)
                v = float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))))
                out[c] = {"measure": "cramers_v", "effect": v, "p_value": float(p)}
    else:
        y = pd.to_numeric(d[target], errors="coerce")
        for c in num_cols:
            ok = d[c].notna() & y.notna()
            if ok.sum() > 3 and d.loc[ok, c].std() > 0:
                r, p = st.pearsonr(d.loc[ok, c].astype(float), y[ok])
                out[c] = {"measure": "pearson_r", "effect": float(abs(r)), "signed": float(r), "p_value": float(p)}
        for c in cat_cols:
            groups = [y[d[c].astype(str) == g].dropna().values for g in d[c].astype(str).unique()]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) >= 2:
                f, p = st.f_oneway(*groups)
                allv = np.concatenate(groups)
                ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
                ss_t = ((allv - allv.mean()) ** 2).sum() or 1.0
                out[c] = {"measure": "eta_squared", "effect": float(ss_b / ss_t), "p_value": float(p)}
    return jsonable({"columns": out})


def distribution_fit(df: pd.DataFrame, cols: list[str], seed: int = 0) -> dict:
    """Best-fit candidate (normal, lognormal, exponential, gamma) per feature by KS statistic."""
    out = {}
    rng = np.random.default_rng(seed)
    for c in cols:
        x = df[c].dropna().astype(float).values
        if len(x) < 20 or np.std(x) == 0:
            continue
        if len(x) > 2000:
            x = rng.choice(x, 2000, replace=False)
        cands = {"norm": st.norm}
        if x.min() > 0:
            cands.update({"lognorm": st.lognorm, "gamma": st.gamma})
        if x.min() >= 0:
            cands["expon"] = st.expon
        fits = {}
        for name, dist in cands.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params = dist.fit(x, floc=0) if name in ("lognorm", "gamma", "expon") else dist.fit(x)
                ks = st.kstest(x, dist(*params).cdf).statistic
                fits[name] = float(ks)
            except Exception:  # noqa: BLE001 - a failed fit just drops that candidate
                continue
        if fits:
            best = min(fits, key=fits.get)
            out[c] = {"best": best, "ks": fits[best], "all": fits}
    return jsonable({"columns": out})


def correlation_report(df: pd.DataFrame, cols: list[str], top: int = 10) -> dict:
    """Pearson and Spearman, with the strongest pairs listed."""
    cols = cols[:40]
    if len(cols) < 2:
        return {"top_pairs": [], "mean_abs_corr": 0.0}
    X = df[cols].astype(float)
    pear, spear = X.corr(), X.corr(method="spearman")
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = pear.loc[a, b]
            if pd.notna(r):
                pairs.append({"a": a, "b": b, "pearson": float(r), "spearman": float(spear.loc[a, b])})
    pairs.sort(key=lambda p: -abs(p["pearson"]))
    mean_abs = float(np.mean([abs(p["pearson"]) for p in pairs])) if pairs else 0.0
    return jsonable({"top_pairs": pairs[:top], "mean_abs_corr": mean_abs,
                     "matrix": {"columns": cols, "values": pear.round(3).values.tolist()}})


def label_noise_report(df: pd.DataFrame, target: str, seed: int = 0, max_rows: int = 0) -> dict:
    """Confident-learning-lite: rows whose out-of-fold prediction confidently disagrees with the label.
    With max_rows (the host's row cap) the check runs on a stratified sample of that size; suspect rows are
    reported by their row number in the file either way."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    d = df.reset_index(drop=True)
    d = d[d[target].notna()]
    if max_rows and len(d) > max_rows:
        from sklearn.model_selection import train_test_split
        counts_all = d[target].astype(str).value_counts()
        strat = d[target].astype(str) if counts_all.min() >= 2 else None
        d, _ = train_test_split(d, train_size=max_rows, stratify=strat, random_state=seed)
        d = d.sort_index()
    rows = d.index.to_numpy()
    d = d.reset_index(drop=True)
    y = pd.factorize(d[target].astype(str), sort=True)[0]
    counts = np.bincount(y)
    if len(counts) < 2 or counts.min() < 5 or len(d) < 50:
        return {"estimated_noise_rate": None, "reason": "too few rows per class"}
    num = numeric_features(d, {target})
    cat = categorical_features(d, {target})
    if not num and not cat:
        return {"estimated_noise_rate": None, "reason": "no usable features"}
    prep = ColumnTransformer([("n", SimpleImputer(strategy="median"), num),
                              ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                              ("o", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat)])
    model = Pipeline([("p", prep), ("m", HistGradientBoostingClassifier(max_iter=150, random_state=seed))])
    k = int(min(5, counts.min()))
    proba = cross_val_predict(model, d[num + cat], y, cv=StratifiedKFold(k, shuffle=True, random_state=seed), method="predict_proba")
    thresholds = np.array([proba[y == j, j].mean() for j in range(len(counts))])
    given = proba[np.arange(len(y)), y]
    other = proba.copy()
    other[np.arange(len(y)), y] = -1
    best_other = other.argmax(axis=1)
    conf_other = other[np.arange(len(y)), best_other]
    suspect = (conf_other >= np.maximum(thresholds[best_other], 0.8)) & (conf_other > given)
    return jsonable({"estimated_noise_rate": float(suspect.mean()), "suspect_count": int(suspect.sum()),
                     "suspect_rows": rows[np.where(suspect)[0][:50]].tolist(), "class_thresholds": thresholds.tolist(),
                     "rows_checked": int(len(d)),
                     "method": "out-of-fold gradient boosting, per-class confidence thresholds"})
