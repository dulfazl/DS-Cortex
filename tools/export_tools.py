"""Exports: the standalone reproduction script, model card, manifest and the export bundle (12A)."""
from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
import zipfile
from pathlib import Path

from app.config import ROOT
from tools import pandas_tools, pipeline_core

LOADER_FUNCS = ["_sniff_delimiter", "load_csv", "_clean_columns", "_parse_ratio_numeric", "_parse_ratio_datetime",
                "infer_kind", "type_summary", "coerce_types"]
PIP_NAMES = {"scikit-learn": "scikit-learn", "pandas": "pandas", "numpy": "numpy", "scipy": "scipy",
             "statsmodels": "statsmodels", "xgboost": "xgboost", "lightgbm": "lightgbm"}


def _lit(obj) -> str:
    """A Python string literal holding the JSON of obj (JSON strings are valid Python literals)."""
    return json.dumps(json.dumps(obj, default=str))


def _strip_future(src: str) -> str:
    return src.replace("from __future__ import annotations\n", "")


def export_pipeline_code(task: dict, cleaning_plan: list[dict], spec: dict, seed: int, data_file: str,
                         reported_score: float | None) -> str:
    """A standalone script: loads the data, rebuilds the pipeline with the same seed and reproduces the
    reported held-out score. It inlines the exact loader and pipeline code DS Cortex ran."""
    loader = "\n\n".join(inspect.getsource(getattr(pandas_tools, f)) for f in LOADER_FUNCS)
    core = _strip_future(inspect.getsource(pipeline_core))
    core = re.sub(r'^"""DS Cortex pipeline core.*?"""\n', "", core, flags=re.S)
    header = f'''#!/usr/bin/env python
"""DS Cortex exported pipeline.

Reproduces the reported held-out score from the raw CSV:

    python pipeline.py --data {data_file}

Steps: load and type the CSV, apply the approved cleaning plan, hold out the same test split
(seed {seed}), fit the chosen pipeline on the training rows, score once on the held-out rows.
Requires: pandas, numpy, scikit-learn (+ xgboost, lightgbm or statsmodels if the model uses them).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TASK = json.loads({_lit(task)})
CLEANING_PLAN = json.loads({_lit(cleaning_plan)})
SPEC = json.loads({_lit(spec)})
SEED = {seed}
REPORTED_SCORE = {reported_score!r}
ID_HINTS = {pandas_tools.ID_HINTS!r}


class DataError(ValueError):
    pass


# ---------------------------------------------------------------- loader (identical to DS Cortex)
'''
    main = '''

# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce the DS Cortex held-out score.")
    ap.add_argument("--data", required=True, help="path to the original CSV")
    args = ap.parse_args()
    df = coerce_types(load_csv(args.data))
    df, changes = apply_cleaning_plan(df, CLEANING_PLAN)
    out = fit_and_score(df, TASK, SPEC, SEED, keep_model=False)
    result = {"metric": TASK["metric"], "score": out["score"], "test": out["test"], "n_train": out["n_train"],
              "n_test": out["n_test"], "reported_score": REPORTED_SCORE}
    if REPORTED_SCORE is not None:
        result["matches_report"] = bool(abs(out["score"] - REPORTED_SCORE) <= 1e-6 + 1e-6 * abs(REPORTED_SCORE))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
'''
    return header + loader + "\n\n# ---------------------------------------------------------------- pipeline core\n" + core + main


def requirements_for(versions: dict, model: str) -> str:
    need = ["pandas", "numpy", "scikit-learn", "scipy"]
    if model == "xgboost":
        need.append("xgboost")
    if model == "lightgbm":
        need.append("lightgbm")
    if model == "arima":
        need.append("statsmodels")
    return "\n".join(f"{PIP_NAMES[n]}=={versions[n]}" if n in versions else PIP_NAMES[n] for n in need) + "\n"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def model_card(s, final_entry_steps: dict, importance: dict | None) -> str:
    plan, test = s.task_plan, s.results["test"]
    fr = s.results["final_recommendation"]
    chosen = s.candidate(s.chosen_id)
    lines = [f"# Model card: {chosen.name}", "",
             "Generated from the Decision Ledger and the held-out results; every number below comes from a logged tool run.", "",
             "## Intended use", "", f"{s.problem_statement}", "",
             "## Data", "", f"- File: {s.file_name} ({s.profile['rows']} rows, {s.profile['columns']} columns)",
             f"- Rows after cleaning: {s.stage_outputs['clean']['rows']} (train {s.stage_outputs['clean']['n_train']}, "
             f"held-out test {s.stage_outputs['clean']['n_test']})",
             f"- Audit: {s.audit_report.summary}"]
    for f in s.audit_report.findings[:5]:
        lines.append(f"  - [{f.severity}] {f.issue_type} {f.column or ''}: {f.rationale}")
    lines += ["", "## Task and metric", "", f"- Task: {plan.task_family}; target: {plan.target or 'none'}",
              f"- Primary metric (fixed before modelling): {plan.metric}; secondary: {', '.join(plan.secondary_metrics) or 'none'}",
              "", "## Performance (held-out test set, used once)", "",
              f"- {plan.metric}: {test['score']:.4f} (baseline {test['baseline']['score']:.4f})"]
    vs = test.get("vs_baseline") or {}
    if vs.get("applicable"):
        lo, hi = vs["ci95"]
        lines.append(f"- Gain over the baseline: {vs['gain']:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}): "
                     + ("more than chance" if vs["meaningful"] else "not more than chance; do not use this model for decisions"))
    if test.get("runner_up"):
        lines.append(f"- Runner-up {test['runner_up']['id']}: {test['runner_up']['score']:.4f}")
    lines += [f"- Confidence in the choice: {fr['confidence']['confidence']:.2f} ({fr['confidence']['formula']})"]
    sub = [o for o in chosen.objections if o.check == "subgroup_check"]
    lines += ["", "## Subgroups", ""]
    lines += [f"- {o.message}" for o in sub] or ["- No subgroup drop above the threshold was flagged for this pipeline."]
    lines += ["", "## Assumptions and limits", ""]
    lines += [f"- {fl.issue}: {fl.detail}" for fl in s.stats_profile.flags] or ["- No assumption violations flagged."]
    lines += ["- Drift: use the drift check (alert at drift score 0.3) on new batches; it does not retrain or monitor.",
              "- Not for: decisions about individuals without human review; data unlike the training data."]
    if importance:
        if vs.get("meaningful") is False:
            lines += ["", "## Feature importance", "", "Not drivers: the model does not beat the baseline by more than chance, "
                      "so these numbers describe noise.", ""]
        else:
            lines += ["", "## Main drivers", ""]
        lines += [f"- {k}: {v:.4f}" for k, v in list(importance["importance"].items())[:8]]
        if importance.get("substituted"):
            lines.append(f"- Note: {importance['substitution_reason']}; permutation importance used instead.")
    return "\n".join(lines) + "\n"


def write_zip(files: dict[str, Path | bytes | str], out: Path) -> Path:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, src in files.items():
            if isinstance(src, Path):
                z.write(src, name)
            else:
                z.writestr(name, src if isinstance(src, (bytes, str)) else str(src))
    return out


def safe_csv_cell(v) -> str:
    """Prefix cells that spreadsheet apps would run as formulas."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s
