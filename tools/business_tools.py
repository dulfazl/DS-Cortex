"""Facts the Business Insight agent reasons over: interpretability, cost and closeness to the best."""
from __future__ import annotations

from tools import pipeline_core as pc
from tools.common import jsonable


def candidate_table(cands: list[dict], metric: str) -> dict:
    best = max(cands, key=lambda c: pc.oriented(metric, c["mean"]))
    out = {}
    for c in cands:
        gap = pc.oriented(metric, best["mean"]) - pc.oriented(metric, c["mean"])
        out[c["id"]] = {"model": c["model"], "interpretable": c["interpretable"], "fit_time": c["fit_time"],
                        "primary_mean": c["mean"], "primary_std": c["std"], "gap_to_best": gap,
                        "within_1sd": bool(gap <= best["std"])}
    return jsonable({"metric": metric, "best": best["id"], "best_std": best["std"], "candidates": out})


def kpis(metric: str, score: float, baseline: float) -> dict:
    """Gain over the baseline, oriented so positive is better, absolute and relative."""
    gain = pc.oriented(metric, score) - pc.oriented(metric, baseline)
    rel = gain / abs(baseline) if baseline else None
    return jsonable({"metric": metric, "score": score, "baseline": baseline, "gain": gain, "relative_gain": rel,
                     "relative_gain_pct": None if rel is None else round(100 * rel, 1)})
