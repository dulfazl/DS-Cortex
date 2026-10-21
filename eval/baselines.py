"""External baselines for the ablation, scored on the SAME held-out split and metric:
  A. single-LLM: one Gemini call writes a pipeline from the prompt; it runs in a separate process
  B. AutoML: FLAML with default settings and a fixed time budget"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.llm import BaseLLM
from tools import pipeline_core as pc
from tools.pandas_tools import coerce_types, load_csv, quick_profile

FLAML_METRIC = {"f1_macro": "macro_f1", "pr_auc": "ap", "rmse": "rmse", "roc_auc": "roc_auc"}


def _split(path: Path, task: dict, seed: int):
    df = coerce_types(load_csv(path))
    df = df[df[task["target"]].notna()].reset_index(drop=True)
    tr, te = pc.holdout_split(df, task, seed)
    return df, tr, te


def automl_flaml(path: Path, task: dict, seed: int, budget: int = 60) -> dict:
    if task["task_family"] not in ("classification", "regression"):
        return {"status": "not_applicable", "reason": f"FLAML baseline covers classification and regression, not {task['task_family']}"}
    from flaml import AutoML
    df, tr, te = _split(path, task, seed)
    target = task["target"]
    X = df.drop(columns=[target])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category")
    y = df[target]
    if task["task_family"] == "classification":
        classes, y_enc = np.unique(y.astype(str), return_inverse=True)
    else:
        classes, y_enc = np.array([]), y.astype(float).values
    automl = AutoML()
    t0 = time.perf_counter()
    automl.fit(X.iloc[tr], y_enc[tr], task=task["task_family"], metric=FLAML_METRIC.get(task["metric"], "auto"),
               time_budget=budget, seed=seed, verbose=0)
    pred = automl.predict(X.iloc[te])
    proba = automl.predict_proba(X.iloc[te]) if task["task_family"] == "classification" else None
    pos = int(np.argmin(np.bincount(y_enc[tr]))) if task["task_family"] == "classification" and len(classes) == 2 else None
    scores = pc.supervised_scores([task["metric"]] + task.get("secondary_metrics", []), y_enc[te], pred, proba, classes, pos)
    return {"status": "ok", "score": scores[task["metric"]], "test": scores, "model": str(automl.best_estimator),
            "seconds": time.perf_counter() - t0}


SINGLE_LLM_PROMPT = """You are a data scientist. Write Python that builds a scikit-learn model for this task.
Problem: {statement}
Task family: {family}; target column: {target}; metric: {metric}
Columns (name, kind, missing %): {columns}
Write ONLY a Python function `build(X_columns_numeric, X_columns_categorical)` that returns an UNFITTED
sklearn Pipeline (preprocessing + estimator) using only scikit-learn. No prints, no file access."""

RUNNER = r'''
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[4])
from tools import pipeline_core as pc
from tools.pandas_tools import coerce_types, load_csv
task = json.loads(open(sys.argv[2]).read()); seed = int(sys.argv[3])
ns = {}
exec(open(sys.argv[1]).read(), ns)
df = coerce_types(load_csv(task["path"]))
df = df[df[task["target"]].notna()].reset_index(drop=True)
tr, te = pc.holdout_split(df, task, seed)
info = pc.prepare_supervised(df.iloc[tr].reset_index(drop=True), task)
test = df.iloc[te].reset_index(drop=True)
X_te = test[list(info["X"].columns)]
num = info["cols"]["numeric"]; cat = info["cols"]["categorical"] + info["cols"]["high_cardinality"]
model = ns["build"](num, cat)
model.fit(info["X"], info["y"])
pred = model.predict(X_te)
if task["task_family"] == "classification":
    y_te = np.searchsorted(info["classes"], test[task["target"]].astype(str).values)
    proba = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
else:
    y_te = test[task["target"]].astype(float).values; proba = None
s = pc.supervised_scores([task["metric"]], y_te, pred, proba, info["classes"], info["pos"])
print(json.dumps({"score": s[task["metric"]]}))
'''


def single_llm(path: Path, task: dict, statement: str, seed: int, llm: BaseLLM, timeout: int = 300) -> dict:
    """Config A. The model-written code runs in a separate Python process with a timeout."""
    if not llm.available:
        return {"status": "skipped", "reason": "requires GOOGLE_API_KEY (config A uses one Gemini call)"}
    if task["task_family"] not in ("classification", "regression"):
        return {"status": "not_applicable", "reason": "single-LLM baseline covers classification and regression"}
    prof = quick_profile(path)
    cols = [(c["name"], c["kind"], c["missing_pct"]) for c in prof["column_info"]]
    text = llm.generate(SINGLE_LLM_PROMPT.format(statement=statement, family=task["task_family"], target=task["target"],
                                                 metric=task["metric"], columns=cols))
    if not text:
        return {"status": "failed", "reason": "no reply from the model"}
    code = text.split("```python")[-1].split("```")[0] if "```" in text else text
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "model.py").write_text(code)
        (Path(d) / "task.json").write_text(json.dumps({**task, "path": str(path)}))
        (Path(d) / "runner.py").write_text(RUNNER)
        t0 = time.perf_counter()
        try:
            r = subprocess.run([sys.executable, "runner.py", "model.py", "task.json", str(seed), str(root)], cwd=d,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "failed", "reason": "timeout"}
        if r.returncode != 0:
            return {"status": "failed", "reason": r.stderr[-400:], "code": code}
        out = json.loads(r.stdout.strip().splitlines()[-1])
    return {"status": "ok", "score": out["score"], "seconds": time.perf_counter() - t0, "code": code,
            "calls": llm.usage["calls"], "tokens": llm.usage["prompt_tokens"] + llm.usage["output_tokens"]}
