"""Flaw injection: for each clean supervised dataset, variants with exactly one known flaw each.
The ground truth is recorded so the catch rate (and the false-alarm rate on clean data) can be computed."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FLAWS = ["target_leakage", "proxy_feature", "label_noise", "mnar_missing", "duplicates_across_split", "covariate_shift"]


@dataclass
class Variant:
    dataset: str
    flaw: str
    df: pd.DataFrame
    column: str | None          # the column the flaw lives in (ground truth)
    new_batch: pd.DataFrame | None = None   # covariate shift: the shifted batch for the drift check
    detail: str = ""


def _numeric_target(df: pd.DataFrame, target: str) -> np.ndarray:
    y = df[target]
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(float).values
    return pd.factorize(y.astype(str), sort=True)[0].astype(float)


def _key_feature(df: pd.DataFrame, target: str) -> str | None:
    """The numeric feature most correlated with the target (the one a flaw would hurt most)."""
    y = _numeric_target(df, target)
    best, best_r = None, 0.0
    for c in df.columns:
        if c == target or not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() < 10:
            continue
        r = abs(pd.Series(df[c].values, dtype=float).corr(pd.Series(y)))
        if r == r and r > best_r:
            best, best_r = c, r
    return best


def make_variants(name: str, df: pd.DataFrame, target: str, family: str, seed: int = 0) -> list[Variant]:
    rng = np.random.default_rng(seed)
    y = _numeric_target(df, target)
    sd = float(np.nanstd(y)) or 1.0
    out: list[Variant] = []
    # 1 target leakage: a column equal to the target plus small noise
    d = df.copy()
    d["post_outcome_score"] = y + rng.normal(0, 0.02 * sd, len(d))
    out.append(Variant(name, "target_leakage", d, "post_outcome_score", detail="target + N(0, 0.02 sd)"))
    # 2 proxy feature: a noisy copy of the target
    d = df.copy()
    d["proxy_signal"] = y + rng.normal(0, 0.35 * sd, len(d))
    out.append(Variant(name, "proxy_feature", d, "proxy_signal", detail="target + N(0, 0.35 sd)"))
    # 3 label noise: flip 10 percent of labels (classification only)
    if family == "classification":
        d = df.copy()
        idx = rng.choice(len(d), int(0.1 * len(d)), replace=False)
        labels = d[target].dropna().unique()
        d.loc[idx, target] = [rng.choice([lab for lab in labels if lab != v]) if len(labels) > 1 else v for v in d.loc[idx, target]]
        out.append(Variant(name, "label_noise", d, target, detail="10% of labels flipped"))
    # 4 missing-not-at-random values in a key feature: the top values go missing
    key = _key_feature(df, target)
    if key:
        d = df.copy()
        hi = d[key] > d[key].quantile(0.7)
        drop = hi & (rng.random(len(d)) < 0.8)
        d.loc[drop, key] = np.nan
        out.append(Variant(name, "mnar_missing", d, key, detail=f"80% of values above the 70th percentile of {key} removed"))
    # 5 duplicate rows that would cross the train/test split
    d = pd.concat([df, df.sample(frac=0.1, random_state=seed)], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.append(Variant(name, "duplicates_across_split", d, None, detail="10% of rows duplicated and shuffled in"))
    # 6 covariate shift between training data and a new batch (checked with the drift tool)
    if key:
        batch = df.sample(frac=0.3, random_state=seed + 1).copy()
        batch[key] = batch[key] + 1.0 * float(df[key].std())
        out.append(Variant(name, "covariate_shift", df.copy(), key, new_batch=batch, detail=f"{key} shifted by one sd in the new batch"))
    return out


def caught(flaw: str, column: str | None, audit: dict | None, candidates: list[dict], drift: dict | None) -> bool:
    """Ground-truth check: did the system flag this flaw where it lives?"""
    findings = (audit or {}).get("findings", [])

    def has(issue_types, col=None, severities=None):
        return any(f["issue_type"] in issue_types and (col is None or f["column"] == col)
                   and (severities is None or f["severity"] in severities) for f in findings)

    critic_leak = any(o["check"] == "leakage_probe" for c in candidates for o in c.get("objections", []))
    if flaw == "target_leakage":
        return has({"target_leakage"}, column) or critic_leak
    if flaw == "proxy_feature":
        return has({"target_leakage", "possible_proxy", "suspicious_name"}, column) or critic_leak
    if flaw == "label_noise":
        return has({"label_noise"})
    if flaw == "mnar_missing":
        return any(f["issue_type"] == "missing_values" and f["column"] == column and "MCAR" not in f["rationale"] for f in findings)
    if flaw == "duplicates_across_split":
        return has({"duplicates"})
    if flaw == "covariate_shift":
        return bool(drift and column in drift.get("drifted", []))
    return False
