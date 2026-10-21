"""Loading, profiling and data-quality tools. Every function returns JSON-ready dicts
with executed results; agents copy these into Evidence objects, never paraphrase them."""
from __future__ import annotations

import csv
import io
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.common import DataError, jsonable

TARGET_HINTS = ("target", "label", "class", "outcome", "churn", "default", "survived", "fraud",
                "price", "sales", "revenue", "score", "y", "result", "response", "converted")
ID_HINTS = ("id", "uuid", "key", "index", "code", "number", "no")


# --------------------------------------------------------------------------- loading
def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a CSV robustly: BOM, Latin-1, semicolon/tab delimiters, blank or duplicate headers."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"File not found: {path.name}")
    raw = path.read_bytes()[:65536]
    if not raw.strip():
        raise DataError("The file is empty.")
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:  # pragma: no cover - latin-1 decodes anything
        raise DataError("Could not decode the file. Save it as UTF-8 and try again.")
    sep = _sniff_delimiter(text)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(path, sep=sep, encoding=encoding, nrows=nrows, low_memory=False)
    except pd.errors.EmptyDataError as e:
        raise DataError("The file has no columns.") from e
    except pd.errors.ParserError as e:
        raise DataError(f"The CSV could not be parsed: {str(e).splitlines()[0]}") from e
    if df.shape[1] == 0:
        raise DataError("The file has no columns.")
    if df.shape[0] == 0:
        raise DataError("The file has a header but no data rows.")
    df.columns = _clean_columns(df.columns)
    return df


def _clean_columns(cols) -> list[str]:
    out, seen = [], {}
    for i, c in enumerate(cols):
        name = str(c).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{i}"
        name = re.sub(r"\.\d+$", "", name) if re.search(r"\.\d+$", name) and name.rsplit(".", 1)[0] in seen else name
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def describe_csv(path: str) -> dict:
    """Shape, dtypes and missing counts of a CSV file (the Phase 0 tool)."""
    df = load_csv(path)
    return jsonable({
        "rows": df.shape[0],
        "columns": df.shape[1],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in df.isna().sum().items()},
    })


# --------------------------------------------------------------------------- type inference
def _parse_ratio_numeric(s: pd.Series) -> float:
    vals = s.dropna().astype(str).str.strip().str.replace(",", "", regex=False)
    if vals.empty:
        return 0.0
    return float(pd.to_numeric(vals, errors="coerce").notna().mean())


def _parse_ratio_datetime(s: pd.Series) -> float:
    vals = s.dropna().astype(str).str.strip()
    if vals.empty:
        return 0.0
    if vals.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.5:
        return 0.0  # plain numbers are not dates
    sample = vals.sample(min(len(vals), 500), random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def infer_kind(s: pd.Series) -> dict:
    """Return the semantic kind of a column plus the facts behind the decision."""
    n = len(s)
    non_null = s.dropna()
    nunique = int(non_null.nunique())
    unique_ratio = nunique / max(1, len(non_null))
    info: dict[str, Any] = {"n_unique": nunique, "unique_ratio": round(unique_ratio, 4), "stored_as_text": None}
    name = str(s.name).lower()
    if len(non_null) == 0:
        return {**info, "kind": "empty"}
    if nunique == 1:
        return {**info, "kind": "constant"}
    if pd.api.types.is_bool_dtype(s):
        return {**info, "kind": "boolean"}
    if pd.api.types.is_datetime64_any_dtype(s):
        return {**info, "kind": "datetime"}
    if pd.api.types.is_numeric_dtype(s):
        is_int = pd.api.types.is_integer_dtype(s) or bool(np.all(np.mod(non_null, 1) == 0))
        if nunique == 2:
            return {**info, "kind": "boolean"}
        name_id = any(name == h or name.endswith("_" + h) or name.startswith(h + "_") for h in ID_HINTS)
        if is_int and len(non_null) >= 20 and ((name_id and unique_ratio > 0.9) or
                                               (unique_ratio > 0.98 and non_null.is_monotonic_increasing)):
            return {**info, "kind": "id_like"}
        return {**info, "kind": "numeric"}
    # text-like storage
    str_vals = non_null.astype(str).str.strip()
    lowered = set(str_vals.str.lower().unique()[:10])
    if nunique == 2 or (nunique <= 2 and lowered <= {"yes", "no", "true", "false", "y", "n", "0", "1"}):
        return {**info, "kind": "boolean"}
    if _parse_ratio_numeric(non_null) >= 0.95:
        return {**info, "kind": "numeric", "stored_as_text": "numeric"}
    if _parse_ratio_datetime(non_null) >= 0.9:
        return {**info, "kind": "datetime", "stored_as_text": "datetime"}
    avg_len = float(str_vals.str.len().mean())
    avg_words = float(str_vals.str.split().str.len().mean())
    if avg_len > 40 and avg_words > 5:
        return {**info, "kind": "text"}
    if unique_ratio > 0.95 and len(non_null) >= 20:
        return {**info, "kind": "id_like"}   # codes, names, ticket numbers: identifiers, not features
    return {**info, "kind": "categorical"}


def type_summary(df: pd.DataFrame) -> dict[str, dict]:
    return {c: infer_kind(df[c]) for c in df.columns}


def coerce_types(df: pd.DataFrame, kinds: dict[str, dict] | None = None) -> pd.DataFrame:
    """Convert numerics/dates stored as text into real dtypes (stateless, safe before splitting)."""
    kinds = kinds or type_summary(df)
    out = df.copy()
    for c, k in kinds.items():
        if c not in out:
            continue
        if k.get("stored_as_text") == "numeric":
            out[c] = pd.to_numeric(out[c].astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce")
        elif k.get("stored_as_text") == "datetime" or (k["kind"] == "datetime" and not pd.api.types.is_datetime64_any_dtype(out[c])):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[c] = pd.to_datetime(out[c], errors="coerce", format="mixed")
    return out


# --------------------------------------------------------------------------- quick profile
def quick_profile(path: str | Path, max_rows: int = 200_000) -> dict:
    """Fast first look: counts, inferred kinds, missing share, cardinality, candidate targets."""
    path = Path(path)
    df = load_csv(path)
    n_rows = len(df)
    sampled = n_rows > max_rows
    if sampled:
        df = df.sample(max_rows, random_state=0)
    kinds = type_summary(df)
    cols = []
    for c in df.columns:
        k = kinds[c]
        ex = [str(v)[:40] for v in df[c].dropna().unique()[:3]]
        cols.append({
            "name": c, "kind": k["kind"], "dtype": str(df[c].dtype),
            "missing_pct": round(100 * float(df[c].isna().mean()), 2),
            "n_unique": k["n_unique"], "unique_ratio": k["unique_ratio"],
            "stored_as_text": k.get("stored_as_text"), "examples": ex,
        })
    return jsonable({
        "file": path.name,
        "file_size_mb": round(path.stat().st_size / 1e6, 3),
        "rows": n_rows,
        "columns": df.shape[1],
        "sampled": sampled,
        "sample_rows": len(df),
        "column_info": cols,
        "candidate_targets": candidate_targets(cols),
        "datetime_columns": [c["name"] for c in cols if c["kind"] == "datetime"],
        "missing_pct_total": round(100 * float(df.isna().mean().mean()), 2),
    })


def candidate_targets(cols: list[dict]) -> list[str]:
    scored = []
    for i, c in enumerate(cols):
        if c["kind"] in ("id_like", "constant", "empty", "text", "datetime"):
            continue
        name = c["name"].lower()
        score = 0.0
        if any(name == h or name.startswith(h + "_") or name.endswith("_" + h) or name.startswith("is_") for h in TARGET_HINTS):
            score += 2
        if c["kind"] == "boolean":
            score += 1
        if i == len(cols) - 1:
            score += 1
        if score > 0:
            scored.append((score, c["name"]))
    return [n for _, n in sorted(scored, key=lambda t: -t[0])][:5]


def read_text_preview(path: str | Path, n: int = 5) -> str:
    df = load_csv(path, nrows=n)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# =========================================================================== audit tools (Phase 3)
FUTURE_NAME_HINTS = ("outcome", "result", "cancel", "refund", "closed", "final", "after", "post_", "_post",
                     "resolution", "settled", "paid_off", "churned", "approved", "status_end", "end_date",
                     "days_to_", "label", "target", "leak")
NONNEG_HINTS = ("age", "price", "amount", "count", "quantity", "qty", "salary", "income", "weight", "height",
                "distance", "duration", "years", "months", "tenure", "fare", "cost", "revenue", "sales")
UNIT_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*([a-zA-Z%]{1,4})\s*$")


def _usable_numeric(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique(dropna=True) > 2]


def missing_report(df: pd.DataFrame, target: str | None = None) -> dict:
    """Percent missing per column, plus a mechanism guess that is never stated as certain."""
    from scipy import stats as st
    out = {}
    n = len(df)
    for col in df.columns:
        miss = df[col].isna()
        k = int(miss.sum())
        if k == 0:
            continue
        rec: dict[str, Any] = {"missing_pct": round(100 * k / n, 3), "missing_count": k, "associations": []}
        tests = 0
        best_p, best_eff, best_with = 1.0, 0.0, None
        if 0 < k < n:
            m = miss.astype(int).values
            for other in df.columns:
                if other == col:
                    continue
                s = df[other]
                ok = s.notna().values
                if ok.sum() < 20:
                    continue
                if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2:
                    mm, vv = m[ok], s.values[ok].astype(float)
                    if mm.min() == mm.max() or np.std(vv) == 0:
                        continue
                    r, p = st.pointbiserialr(mm, vv)
                    eff = abs(r)
                elif s.nunique() <= 30:
                    tab = pd.crosstab(m[ok], s.values[ok])
                    if tab.shape[0] < 2 or tab.shape[1] < 2:
                        continue
                    chi2, p, _, _ = st.chi2_contingency(tab)
                    eff = float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))))
                else:
                    continue
                tests += 1
                if p < best_p:
                    best_p, best_eff, best_with = float(p), float(eff), other
            adj_p = min(1.0, best_p * max(1, tests))
            rec["associations"] = {"strongest_with": best_with, "p_adjusted": adj_p, "effect": round(best_eff, 4), "tests": tests}
            # MNAR hint: do rows with a missing value look different on what the column itself would be?
            self_d, tail = None, None
            num_others = [c for c in _usable_numeric(df, {col, target} if target else {col}) if df[c].notna().mean() > 0.8]
            if pd.api.types.is_numeric_dtype(df[col]) and num_others and (n - k) > 30 and k >= 5:
                from sklearn.linear_model import LinearRegression
                Xo = df[num_others].fillna(df[num_others].median())
                lr = LinearRegression().fit(Xo[~miss], df.loc[~miss, col])
                r2 = lr.score(Xo[~miss], df.loc[~miss, col])
                if r2 > 0.2:
                    pred = lr.predict(Xo)
                    a, b = pred[miss.values], pred[~miss.values]
                    pooled = np.sqrt((a.var() + b.var()) / 2) or 1.0
                    self_d = float(abs(a.mean() - b.mean()) / pooled)
                    obs = df.loc[~miss, col]
                    if np.median(a) > obs.quantile(0.75):
                        tail = "upper"
                    elif np.median(a) < obs.quantile(0.25):
                        tail = "lower"
            rec["self_effect_d"] = self_d
            rec["tail"] = tail
            # MNAR cannot be tested from observed data. "possibly MNAR" only when the rows with a missing value
            # are predicted to sit in one tail of the column (e.g. high incomes left blank); confidence stays modest.
            if self_d is not None and self_d >= 0.8 and tail:
                rec["mechanism"] = "possibly MNAR"
                rec["confidence"] = round(min(0.75, 0.5 + 0.25 * min(1.0, (self_d - 0.8) / 1.0)), 3)
            elif adj_p < 0.01 and best_eff >= 0.1:
                rec["mechanism"] = "MAR-like"
                rec["confidence"] = round(min(0.95, 0.5 + best_eff), 3)
            else:
                rec["mechanism"] = "MCAR-like"
                rec["confidence"] = round(0.5 + 0.3 * min(1.0, adj_p), 3)
        else:
            rec["mechanism"] = "entirely missing"
            rec["confidence"] = 1.0
        out[col] = rec
    return jsonable({"columns": out, "n_rows": n, "note": "mechanism labels are guesses from observed data, never certain"})


def type_inference(df: pd.DataFrame) -> dict:
    """Id-like columns, constants, dates or numbers stored as text, high-cardinality categoricals."""
    res = {}
    for c in df.columns:
        k = infer_kind(df[c])
        issue = None
        if k["kind"] == "id_like":
            issue = "id_like"
        elif k["kind"] in ("constant", "empty"):
            issue = k["kind"]
        elif k.get("stored_as_text"):
            issue = f"{k['stored_as_text']}_stored_as_text"
        elif k["kind"] == "categorical" and (k["n_unique"] > 30):
            issue = "high_cardinality"
        elif k["kind"] == "text":
            issue = "free_text"
        res[c] = {**k, "issue": issue}
    return jsonable({"columns": res})


def outlier_report(df: pd.DataFrame, exclude: list[str] | None = None, seed: int = 0) -> dict:
    """IQR and z-score per numeric column, plus Isolation Forest on the numeric block."""
    from sklearn.ensemble import IsolationForest
    cols = _usable_numeric(df, set(exclude or []))
    out: dict[str, Any] = {"columns": {}, "isolation_forest": None}
    if not cols:
        return out
    block = df[cols].astype(float)
    filled = block.fillna(block.median())
    if_flags = np.zeros(len(df), dtype=bool)
    if len(df) >= 20:
        std = (filled - filled.mean()) / filled.std(ddof=0).replace(0, 1)
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed).fit(std)
        if_flags = iso.predict(std) == -1
        out["isolation_forest"] = {"flagged_rows": int(if_flags.sum()), "flagged_pct": round(100 * float(if_flags.mean()), 3)}
    for c in cols:
        s = block[c]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_mask = ((s < lo) | (s > hi)).fillna(False).values
        sd = s.std(ddof=0)
        z_mask = (((s - s.mean()).abs() / (sd if sd else 1)) > 3).fillna(False).values
        n_iqr = int(iqr_mask.sum())
        agree = 0.0
        if n_iqr:
            agree = (float((iqr_mask & z_mask).sum()) + float((iqr_mask & if_flags).sum())) / (2 * n_iqr)
        out["columns"][c] = {"iqr_count": n_iqr, "iqr_pct": round(100 * n_iqr / max(1, s.notna().sum()), 3),
                             "z_count": int(z_mask.sum()), "if_overlap": int((iqr_mask & if_flags).sum()),
                             "agreement": round(agree, 4), "fences": [float(lo), float(hi)],
                             "skew": round(float(s.skew()), 4) if s.notna().sum() > 2 else None}
    return jsonable(out)


def duplicate_report(df: pd.DataFrame, key_columns: list[str] | None = None) -> dict:
    """Exact duplicates, and near-duplicates on key columns (case/space/rounding-insensitive)."""
    exact = df.duplicated(keep="first")
    kinds = type_summary(df)
    keys = key_columns or [c for c in df.columns if kinds[c]["kind"] not in ("id_like",)]
    norm = df[keys].copy()
    for c in keys:
        if pd.api.types.is_numeric_dtype(norm[c]):
            norm[c] = norm[c].round(6)
        else:
            norm[c] = norm[c].astype(str).str.strip().str.lower()
    near = norm.duplicated(keep="first") & ~exact
    return jsonable({"exact_count": int(exact.sum()), "exact_pct": round(100 * float(exact.mean()), 3),
                     "exact_examples": np.where(exact)[0][:10].tolist(), "near_count": int(near.sum()),
                     "near_pct": round(100 * float(near.mean()), 3), "near_examples": np.where(near)[0][:10].tolist(),
                     "key_columns": keys})


def imbalance_report(df: pd.DataFrame, target: str) -> dict:
    vc = df[target].astype(str).value_counts()
    shares = vc / vc.sum()
    return jsonable({"target": target, "counts": vc.to_dict(), "shares": shares.round(4).to_dict(),
                     "minority_class": str(shares.idxmin()), "minority_share": float(shares.min()),
                     "imbalance_ratio": float(vc.max() / vc.min())})


def leakage_report(df: pd.DataFrame, target: str, task_family: str, seed: int = 0) -> dict:
    """Features that predict the target almost perfectly on their own, or whose names suggest
    information from after the prediction time."""
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    d = df[df[target].notna()]
    y_raw = d[target]
    is_clf = task_family in ("classification", "anomaly_detection")
    if is_clf:
        y = pd.factorize(y_raw.astype(str))[0]
        base = float(pd.Series(y).value_counts(normalize=True).max())
    else:
        y = pd.to_numeric(y_raw, errors="coerce").values
    feats = {}
    for c in d.columns:
        if c == target:
            continue
        s = d[c]
        rec: dict[str, Any] = {}
        name = c.lower()
        rec["name_flag"] = next((h for h in FUTURE_NAME_HINTS if h in name), None)
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 1:
            x = s.fillna(s.median()).values.astype(float).reshape(-1, 1)
            if not is_clf or len(np.unique(y)) == 2:
                rec["pearson"] = float(abs(pd.Series(x[:, 0]).corr(pd.Series(y.astype(float)))))
                rec["spearman"] = float(abs(pd.Series(x[:, 0]).corr(pd.Series(y.astype(float)), method="spearman")))
            if is_clf:
                stump = DecisionTreeClassifier(max_depth=1, random_state=seed)
                acc = float(np.mean(cross_val_score(stump, x, y, cv=3)))
                rec["stump_accuracy"] = acc
                rec["stump_lift"] = float((acc - base) / (1 - base)) if base < 1 else 0.0
                rec["mutual_info"] = float(mutual_info_classif(x, y, random_state=seed)[0])
            else:
                tree = DecisionTreeRegressor(max_depth=3, random_state=seed)
                rec["tree_r2"] = float(np.mean(cross_val_score(tree, x, y, cv=3, scoring="r2")))
        elif s.nunique() > 1:
            g = pd.DataFrame({"x": s.astype(str).values, "y": y})
            n_levels = g["x"].nunique()
            if is_clf and n_levels <= 0.5 * len(g):
                purity = float(g.groupby("x")["y"].agg(lambda v: v.value_counts(normalize=True).max()).mul(
                    g["x"].value_counts(normalize=True).sort_index()).sum())
                rec["category_purity"] = purity
                rec["stump_lift"] = float((purity - base) / (1 - base)) if base < 1 else 0.0
            elif not is_clf and n_levels <= 0.5 * len(g):
                means = g.groupby("x")["y"].transform("mean")
                ss_res = float(((g["y"] - means) ** 2).sum())
                ss_tot = float(((g["y"] - g["y"].mean()) ** 2).sum()) or 1.0
                rec["tree_r2"] = 1 - ss_res / ss_tot
        strength = max(rec.get("stump_lift", 0) or 0, rec.get("pearson", 0) or 0, rec.get("tree_r2", 0) or 0)
        rec["strength"] = round(float(strength), 4)
        if strength >= 0.97:
            rec["severity"] = "blocker"
        elif strength >= 0.85:
            rec["severity"] = "high"
        elif rec["name_flag"] and strength >= 0.3:
            rec["severity"] = "medium"
        else:
            rec["severity"] = None
        feats[c] = rec
    flagged = {c: r for c, r in feats.items() if r["severity"]}
    return jsonable({"target": target, "features": feats, "flagged": sorted(flagged, key=lambda c: -feats[c]["strength"]),
                     "baseline_accuracy": base if is_clf else None})


def inconsistency_report(df: pd.DataFrame) -> dict:
    """Mixed-case categories, impossible values, mixed units."""
    out: dict[str, Any] = {}
    for c in df.columns:
        s = df[c]
        rec: dict[str, Any] = {}
        if not pd.api.types.is_numeric_dtype(s) and s.notna().any():
            vals = s.dropna().astype(str)
            uniq = vals.unique()
            if len(uniq) <= 1000:
                groups = pd.Series(uniq).groupby(pd.Series(uniq).str.strip().str.lower()).size()
                colliding = groups[groups > 1]
                if len(colliding):
                    rec["case_variants"] = {k: [u for u in uniq if u.strip().lower() == k][:5] for k in colliding.index[:5]}
                    rec["case_variant_groups"] = int(len(colliding))
            units = vals.str.extract(UNIT_RE)[0].dropna().str.lower()
            if units.nunique() > 1 and len(units) > 0.5 * len(vals):
                rec["mixed_units"] = units.value_counts().head(5).to_dict()
        elif pd.api.types.is_numeric_dtype(s):
            name = c.lower()
            if any(h in name for h in NONNEG_HINTS):
                neg = int((s < 0).sum())
                if neg:
                    rec["negative_values"] = neg
            if "age" in name.split("_") or name == "age":
                old = int((s > 120).sum())
                if old:
                    rec["impossible_age_over_120"] = old
        if rec:
            out[c] = rec
    return jsonable({"columns": out})
