"""Calibration of the system's confidence. Each run logs (stated confidence, outcome), where the
outcome is: the chosen pipeline beat the baseline AND matched the runner-up within noise on the
held-out data. Bucketed in bands of 0.1; expected calibration error (ECE) and a reliability diagram.
Reported either way: if calibration is poor, the report says so."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def reliability(conf: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        k = int(mask.sum())
        if k == 0:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "mean_confidence": None, "observed_accuracy": None, "gap": None})
            continue
        mc, acc = float(conf[mask].mean()), float(outcome[mask].mean())
        ece += k / n * abs(acc - mc)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": k, "mean_confidence": round(mc, 4), "observed_accuracy": round(acc, 4),
                     "gap": round(acc - mc, 4)})
    return pd.DataFrame(rows), float(ece)


def run(runs: pd.DataFrame, out: Path, configs=("D", "E")) -> dict:
    d = runs[(runs["config"].isin(configs)) & (runs["status"] == "completed") & runs["confidence"].notna()]
    if d.empty:
        return {"n": 0, "ece": None, "note": "no completed runs with a confidence yet"}
    conf = d["confidence"].astype(float).values
    outcome = d["outcome"].astype(bool).astype(float).values
    table, ece = reliability(conf, outcome)
    table.to_csv(out / "calibration.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    t = table.dropna()
    ax.plot([0, 1], [0, 1], ls="--", color="#8a94a6", lw=1, label="perfect calibration")
    ax.bar(t["mean_confidence"], t["observed_accuracy"], width=0.07, color="#3b5bdb", alpha=0.85, label="observed")
    for _, r in t.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_confidence"], r["observed_accuracy"]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7, color="#5d6879")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("observed: recommendation held up on held-out data")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f}, n = {len(d)})", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "reliability.png")
    plt.close(fig)
    verdict = ("well calibrated" if ece < 0.05 else "moderately calibrated" if ece < 0.15 else
               "poorly calibrated: treat the confidence as a ranking, and see future work (temperature scaling)")
    return {"n": int(len(d)), "ece": round(ece, 4), "mean_confidence": round(float(conf.mean()), 4),
            "observed_rate": round(float(outcome.mean()), 4), "verdict": verdict}
