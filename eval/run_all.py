"""One command regenerates every table and figure into reports/eval/:

    python -m eval.run_all --seeds 5            # the pre-registered evaluation (hours on a laptop)
    python -m eval.run_all --quick              # 1 seed, 4 datasets, configs B C D F: a smoke run in minutes
    python -m eval.run_all --report-only        # rebuild tables and figures from the logged runs

Runs are appended to reports/eval/runs.jsonl and flaws.jsonl and are skipped when already present,
so an interrupted evaluation resumes. The evaluation keeps its own database and memory store."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "eval"
STATE = OUT / "cache" / "state"


def _env(fast: bool) -> None:
    os.environ.update(DATA_DIR=str(STATE / "data"), LOG_DIR=str(STATE / "logs"), DB_PATH=str(STATE / "eval.sqlite"),
                      BLOCKER_POLICY="drop", STATIC_PLOTS="0")
    if fast:
        os.environ.update(FAST_MODE="1", CV_REPEATS="1")


def _plots(flaws, runs, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    colors = {"C": "#9aa5b5", "D": "#3b5bdb", "E": "#1f8a5b", "F": "#b26a00"}
    if not flaws.empty and "caught" in flaws:
        fl = flaws[flaws["flaw"] != "clean"].dropna(subset=["caught"])
        if not fl.empty:
            rate = fl.groupby(["flaw", "config"])["caught"].mean().unstack("config")
            fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
            x = np.arange(len(rate.index))
            w = 0.8 / max(1, len(rate.columns))
            for i, c in enumerate(rate.columns):
                ax.bar(x + i * w - 0.4 + w / 2, rate[c].values, w, label=f"config {c}", color=colors.get(c, "#888"))
            ax.set_xticks(x, [f.replace("_", "\n") for f in rate.index], fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("catch rate")
            ax.set_title("Flaw catch rate by flaw type and config", fontsize=10)
            ax.legend(fontsize=8, frameon=False, ncol=4)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            fig.tight_layout()
            fig.savefig(out / "flaw_catch.png")
            plt.close(fig)
    ok = runs[(runs["status"] == "completed") & runs.get("claims", 0).fillna(0).gt(0)] if "claims" in runs else runs.iloc[0:0]
    if not ok.empty:
        h = ok.groupby("config").apply(lambda g: g["rejected"].sum() / g["claims"].sum(), include_groups=False)
        fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=150)
        ax.bar(h.index, h.values, color=[colors.get(c, "#888") for c in h.index])
        for i, v in enumerate(h.values):
            ax.annotate(f"{v:.1%}", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax.set_ylabel("claims rejected by the verifier")
        ax.set_title("Hallucinated-claim rate by config", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "hallucination.png")
        plt.close(fig)


def report(out: Path, seeds_requested: int | None = None) -> dict:
    import pandas as pd
    from eval import calibration_check, stats
    from eval.benchmark_datasets import describe_suite
    runs = pd.read_json(out / "runs.jsonl", lines=True) if (out / "runs.jsonl").exists() else pd.DataFrame()
    flaws = pd.read_json(out / "flaws.jsonl", lines=True) if (out / "flaws.jsonl").exists() else pd.DataFrame()
    summary: dict = {"generated": time.strftime("%Y-%m-%d %H:%M"), "runs": len(runs), "flaw_runs": len(flaws)}
    describe_suite().to_csv(out / "datasets.csv", index=False)
    if not runs.empty:
        ok = runs[runs["status"].isin(["completed", "ok"])]
        ab = ok.groupby(["dataset", "config"])["score"].agg(["mean", "std", "count"]).round(4).reset_index()
        ab["cell"] = ab.apply(lambda r: f"{r['mean']:.4f} ± {0 if pd.isna(r['std']) else r['std']:.4f} (n={int(r['count'])})", axis=1)
        table = ab.pivot(index="dataset", columns="config", values="cell")
        metric = ok.groupby("dataset")["metric"].first()
        table.insert(0, "metric", metric)
        table.to_csv(out / "ablation.csv")
        cost_cols = [c for c in ("seconds", "llm_calls", "tokens") if c in ok]
        ok.groupby("config")[cost_cols].mean().round(2).to_csv(out / "cost.csv")
        st_table = stats.compare(runs)
        st_table.to_csv(out / "stats.csv", index=False)
        summary["calibration"] = calibration_check.run(runs, out)
        if "claims" in ok:
            hc = ok[ok["claims"].fillna(0) > 0].groupby("config")[["claims", "rejected", "claims_llm", "rejected_llm"]].sum()
            hc["hallucinated_claim_rate"] = (hc["rejected"] / hc["claims"]).round(4)
            hc.to_csv(out / "hallucination.csv")
        if "memory" in ok:
            mem = [dict(m, dataset=d, seed=s) for m, d, s in zip(ok.loc[ok["config"] == "E", "memory"], ok.loc[ok["config"] == "E", "dataset"],
                                                                 ok.loc[ok["config"] == "E", "seed"]) if isinstance(m, dict)]
            pd.DataFrame(mem).to_csv(out / "memory.csv", index=False)
        failed = runs[~runs["status"].isin(["completed", "ok", "skipped", "not_applicable"])]
        summary["failed_runs"] = failed[["config", "dataset", "seed", "error"]].to_dict("records") if "error" in failed else []
        summary["skipped"] = runs[runs["status"].isin(["skipped", "not_applicable"])][["config", "dataset", "status"]].drop_duplicates().to_dict("records")
    if not flaws.empty:
        fl = flaws[(flaws["flaw"] != "clean") & flaws.get("caught", pd.Series(dtype=object)).notna()] if "caught" in flaws else flaws.iloc[0:0]
        if not fl.empty:
            fl.groupby(["flaw", "config"])["caught"].agg(["mean", "count"]).round(3).to_csv(out / "flaw_catch.csv")
        clean = flaws[flaws["flaw"] == "clean"]
        fa_cols = [c for c in clean.columns if c.startswith("false_alarm_")]
        if fa_cols:
            clean.groupby("config")[fa_cols].mean().round(3).to_csv(out / "false_alarms.csv")
    _plots(flaws, runs, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _summary_md(out, summary)
    return summary


def _summary_md(out: Path, summary: dict) -> None:
    import pandas as pd
    lines = ["# DS Cortex evaluation results", "", f"Generated {summary['generated']}. Runs logged: {summary['runs']}; "
             f"flaw runs: {summary['flaw_runs']}.", ""]
    for name, title in [("ablation.csv", "Held-out primary metric per dataset and config (mean ± sd over seeds)"),
                        ("stats.csv", "Paired comparisons over datasets (relative gain over baseline)"),
                        ("flaw_catch.csv", "Flaw catch rate"), ("false_alarms.csv", "False-alarm rate on clean data"),
                        ("hallucination.csv", "Hallucinated-claim rate"), ("calibration.csv", "Calibration bins"),
                        ("cost.csv", "Cost per run"), ("memory.csv", "Memory: suggested, entered, won")]:
        p = out / name
        if p.exists():
            df = pd.read_csv(p)
            lines += [f"## {title}", "", df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False), ""]
    cal = summary.get("calibration")
    if cal and cal.get("ece") is not None:
        lines += [f"Calibration: ECE {cal['ece']} over {cal['n']} runs (mean confidence {cal['mean_confidence']}, observed "
                  f"{cal['observed_rate']}): {cal['verdict']}.", ""]
    if summary.get("skipped"):
        lines += ["## Skipped", ""] + [f"- config {s['config']} on {s['dataset']}: {s['status']}" for s in summary["skipped"]] + [""]
    if summary.get("failed_runs"):
        lines += ["## Failed runs (reported, not hidden)", ""] + [f"- {f['config']} {f['dataset']} seed {f['seed']}: {f['error']}"
                                                                 for f in summary["failed_runs"]] + [""]
    (out / "summary.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--configs", default="A,B,C,D,E,F")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full-settings", action="store_true", help="5x3 CV and 20 tuning trials instead of the fast settings")
    ap.add_argument("--no-flaws", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="delete logged runs and the eval memory store first")
    ap.add_argument("--automl-budget", type=int, default=60)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for f in ("runs.jsonl", "flaws.jsonl"):
            (OUT / f).unlink(missing_ok=True)
        shutil.rmtree(STATE, ignore_errors=True)
    _env(fast=not args.full_settings)
    sys.path.insert(0, str(ROOT))
    from app.config import reset_settings
    reset_settings()
    from app.db import init_db
    init_db()
    from eval.ablation_runner import run_ablation, run_flaws
    from eval.benchmark_datasets import SUITE
    if args.quick:
        names = {"titanic", "diabetes", "wine", "airline_passengers"}
        suite = [b for b in SUITE if b.name in names]
        seeds = [42]
        configs = ["B", "C", "D", "F"]
        automl_budget = 20
    else:
        suite = SUITE if args.datasets == "all" else [b for b in SUITE if b.name in set(args.datasets.split(","))]
        seeds = [42 + i for i in range(args.seeds)]
        configs = args.configs.split(",")
        automl_budget = args.automl_budget
    if not args.report_only:
        data_dir = OUT / "cache" / "datasets"
        run_ablation(suite, configs, seeds, OUT, data_dir, automl_budget)
        if not args.no_flaws:
            flaw_configs = [c for c in configs if c in ("C", "D")]
            run_flaws(suite[:2] if args.quick else suite, flaw_configs, seeds[:1] if args.quick else seeds[:2], OUT, data_dir)
    s = report(OUT)
    print(json.dumps({k: v for k, v in s.items() if k != "failed_runs"}, indent=2, default=str))
    print(f"\nTables and figures: {OUT}")


if __name__ == "__main__":
    main()
