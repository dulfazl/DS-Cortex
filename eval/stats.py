"""Paired comparisons of configurations over datasets (Wilcoxon signed-rank, Demsar 2006), with the
rank-biserial effect size and a bootstrap confidence interval of the median difference.
Scores are compared as relative gain over each dataset's baseline, so different metrics share a scale.
With 8 to 10 datasets, results are indicative and reported as such."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

PAIRS = [("D", "C", "deliberation vs sequential"), ("D", "B", "deliberation vs AutoML"), ("D", "A", "deliberation vs single LLM"),
         ("E", "D", "memory on vs off"), ("D", "F", "verification on vs off")]


def per_dataset(runs: pd.DataFrame) -> pd.DataFrame:
    ok = runs[(runs["status"].isin(["completed", "ok"])) & runs["relative_gain"].notna()]
    return ok.groupby(["config", "dataset"])["relative_gain"].mean().unstack("config")


def compare(runs: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    table = per_dataset(runs)
    rows = []
    rng = np.random.default_rng(seed)
    for a, b, what in PAIRS:
        if a not in table or b not in table:
            rows.append({"comparison": f"{a} vs {b}", "question": what, "n_datasets": 0, "note": "config not run"})
            continue
        pair = table[[a, b]].dropna()
        n = len(pair)
        if n < 2:
            rows.append({"comparison": f"{a} vs {b}", "question": what, "n_datasets": n, "note": "fewer than 2 paired datasets"})
            continue
        d = (pair[a] - pair[b]).values
        nz = d[d != 0]
        if len(nz) == 0:
            stat, p = float("nan"), 1.0
        else:
            res = st.wilcoxon(nz)
            stat, p = float(res.statistic), float(res.pvalue)
        ranks = st.rankdata(np.abs(nz)) if len(nz) else np.array([])
        w_plus, w_minus = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
        rbc = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) else 0.0
        boots = [np.median(d[rng.integers(0, n, n)]) for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"comparison": f"{a} vs {b}", "question": what, "n_datasets": n,
                     "median_diff": round(float(np.median(d)), 4), "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
                     "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                     "wilcoxon_W": stat, "p_value": round(p, 4), "rank_biserial": round(rbc, 3),
                     "note": "indicative only: few datasets" if n < 12 else ""})
    return pd.DataFrame(rows)
