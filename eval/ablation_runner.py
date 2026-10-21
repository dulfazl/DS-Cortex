"""Ablation runner: configs A to F over the benchmark suite and seeds, plus the flaw-injection runs.
Every result is appended to runs.jsonl / flaws.jsonl, so an interrupted evaluation resumes where it
stopped."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from app.llm import get_llm
from app.session import RunFlags, store
from eval import baselines
from eval.benchmark_datasets import Benchmark, load, materialize
from eval.inject_flaws import caught, make_variants
from tools import drift_check_tools as dc
from tools import pipeline_core as pc
from tools.tool_log import ToolLog
from tools.verify_tools import verify_claims

CONFIGS: dict[str, dict] = {
    "A": {"label": "Single-LLM baseline", "kind": "single_llm"},
    "B": {"label": "AutoML baseline (FLAML)", "kind": "automl"},
    "C": {"label": "Sequential pipeline", "flags": dict(critic=False, debate=False, verify=False, memory=False)},
    "D": {"label": "Full deliberative", "flags": dict(critic=True, debate=True, verify=True, memory=False)},
    "E": {"label": "Full with memory", "flags": dict(critic=True, debate=True, verify=True, memory=True, memory_exclude_same=True)},
    "F": {"label": "No verification", "flags": dict(critic=True, debate=True, verify=False, memory=False)},
}


def _done(path: Path, keys: tuple[str, ...]) -> set[tuple]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            out.add(tuple(r.get(k) for k in keys))
        except json.JSONDecodeError:
            continue
    return out


def _append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _answers(b: Benchmark) -> dict:
    return {"clarify": {"target": b.target} if b.target and b.family != "anomaly_detection" else
            {"task_family": b.family, **({"target": b.target} if b.target else {})}}


def posthoc_claims(s) -> dict:
    """Hallucinated-claim rate measured after the run with the same verifier, for every config
    (config F used no verification during the run, but its claims are still checkable)."""
    if not s.debate:
        return {"claims": 0, "rejected": 0, "rejected_llm": 0, "claims_llm": 0}
    log = ToolLog(s.id)
    claims = [c.model_copy(deep=True) for p in s.debate.positions for c in p.reasons]
    counts = verify_claims(claims, log)
    return {"claims": counts["total"], "rejected": counts["rejected"], "rejected_llm": counts["rejected_llm"],
            "claims_llm": counts["total_llm"]}


def run_cortex(b: Benchmark, path: Path, config: str, seed: int, extra_flags: dict | None = None) -> tuple[dict, object]:
    from workflows.pipeline_graph import PipelineRun
    flags = RunFlags(**{**CONFIGS[config]["flags"], **(extra_flags or {})})
    s = store.create(str(path), b.statement, mode="autopilot", flags=flags, seed=seed)
    t0 = time.perf_counter()
    s = PipelineRun(s.id, interactive=False, auto_answers=_answers(b)).run()
    secs = time.perf_counter() - t0
    rec = {"status": s.status, "error": s.error, "seconds": round(secs, 2), "session": s.id,
           "llm_calls": s.usage.get("calls", 0), "tokens": s.usage.get("prompt_tokens", 0) + s.usage.get("output_tokens", 0)}
    if s.status == "completed":
        t = s.results["test"]
        fr = s.results["final_recommendation"]
        rec.update(metric=t["metric"], score=t["score"], baseline=t["baseline"]["score"],
                   runner_up=(t.get("runner_up") or {}).get("score"), chosen=s.chosen_id,
                   chosen_model=s.candidate(s.chosen_id).model, confidence=fr["confidence"]["confidence"],
                   outcome=s.results["calibration"]["outcome"], rule=s.debate.rule if s.debate else None,
                   forced_tiebreak=bool(s.debate and s.debate.forced_tiebreak),
                   n_candidates=len(s.candidates), memory=s.memory.get("outcome"), **posthoc_claims(s))
        rec["relative_gain"] = (pc.oriented(t["metric"], t["score"]) - pc.oriented(t["metric"], t["baseline"]["score"])) / (abs(t["baseline"]["score"]) or 1.0)
    return rec, s


def run_ablation(suite: list[Benchmark], configs: list[str], seeds: list[int], out: Path, data_dir: Path,
                 automl_budget: int = 60, log=print) -> None:
    runs = out / "runs.jsonl"
    done = _done(runs, ("config", "dataset", "seed"))
    llm = get_llm()
    # E needs a filled memory store: run D (which stores fingerprints) before E
    order = sorted(configs, key=lambda c: {"D": 0, "E": 2}.get(c, 1))
    for config in order:
        for b in suite:
            path = materialize(b, data_dir)
            for seed in seeds:
                if (config, b.name, seed) in done:
                    continue
                log(f"[{config}] {b.name} seed {seed}")
                base = {"config": config, "label": CONFIGS[config]["label"], "dataset": b.name, "family": b.family, "seed": seed}
                try:
                    kind = CONFIGS[config].get("kind")
                    if kind == "automl":
                        task = _task_for(b)
                        rec = baselines.automl_flaml(path, task, seed, automl_budget)
                        if rec.get("status") == "ok":
                            rec.update(metric=task["metric"], relative_gain=_rel(rec["score"], b, path, task, seed))
                    elif kind == "single_llm":
                        task = _task_for(b)
                        rec = baselines.single_llm(path, task, b.statement, seed, llm)
                        if rec.get("status") == "ok":
                            rec.update(metric=task["metric"], relative_gain=_rel(rec["score"], b, path, task, seed))
                    else:
                        rec, _ = run_cortex(b, path, config, seed)
                except Exception as e:  # noqa: BLE001 - a failed run is a result, recorded as such
                    rec = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
                _append(runs, {**base, **rec})


_TASK_CACHE: dict[str, dict] = {}


def _task_for(b: Benchmark) -> dict:
    """The task plan DS Cortex's router produces for this dataset (baselines must use the same metric)."""
    if b.name in _TASK_CACHE:
        return _TASK_CACHE[b.name]
    from agents.base import AgentContext
    from agents.router.core import RouterAgent
    from eval.benchmark_datasets import CACHE
    from tools.pandas_tools import coerce_types, quick_profile
    path = materialize(b, CACHE.parent / "materialized")
    df = coerce_types(pd.read_csv(path))
    plan = RouterAgent(AgentContext.offline()).plan(b.statement, quick_profile(path), df, answer=_answers(b)["clarify"])
    _TASK_CACHE[b.name] = plan.model_dump()
    return _TASK_CACHE[b.name]


def _rel(score: float, b: Benchmark, path: Path, task: dict, seed: int) -> float:
    from tools.pandas_tools import coerce_types, load_csv
    df = coerce_types(load_csv(path))
    df = df[df[task["target"]].notna()].reset_index(drop=True)
    base = pc.fit_and_score(df, task, pc.baseline_spec(task), seed, keep_model=False)["score"]
    return (pc.oriented(task["metric"], score) - pc.oriented(task["metric"], base)) / (abs(base) or 1.0)


def run_flaws(suite: list[Benchmark], configs: list[str], seeds: list[int], out: Path, data_dir: Path, log=print) -> None:
    """Each supervised dataset x each injected flaw x config. Also runs the clean dataset once per
    config to measure false alarms."""
    path_out = out / "flaws.jsonl"
    done = _done(path_out, ("config", "dataset", "flaw", "seed"))
    for b in [x for x in suite if x.family in ("classification", "regression")]:
        clean = load(b)
        for seed in seeds:
            variants = make_variants(b.name, clean, b.target, b.family, seed)
            for config in configs:
                if config not in ("C", "D", "E", "F"):
                    continue
                for v in [None] + variants:
                    flaw = v.flaw if v else "clean"
                    if (config, b.name, flaw, seed) in done:
                        continue
                    log(f"[flaw {config}] {b.name} {flaw} seed {seed}")
                    df = v.df if v else clean
                    p = data_dir / "flaws" / f"{b.name}__{flaw}__{seed}.csv"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(p, index=False)
                    try:
                        rec, s = run_cortex(b, p, config, seed, extra_flags={"tuning": False, "explain": False})
                        audit = s.audit_report.model_dump() if s.audit_report else None
                        cands = [c.model_dump() for c in s.candidates]
                        drift = None
                        if s.status == "completed" and (flaw in ("covariate_shift", "clean")):
                            batch = v.new_batch if v else clean.sample(frac=0.3, random_state=seed + 1)
                            ref = pd.read_pickle(s.dir / "clean.pkl")
                            drift = dc.drift_check(ref, batch[[c for c in batch.columns if c in ref.columns]], [b.target])
                        if v:
                            hit = caught(v.flaw, v.column, audit, cands, drift)
                            rows = [{"flaw": v.flaw, "caught": hit}]
                        else:   # false alarms: run every detector against the clean data
                            from eval.inject_flaws import FLAWS, _key_feature
                            key = _key_feature(clean, b.target)
                            rows = [{"flaw": "clean", "false_alarm_" + f: caught(f, key if f in ("mnar_missing", "covariate_shift") else None,
                                                                                audit, cands, drift)} for f in FLAWS]
                            rows = [{"flaw": "clean", **{k: v_ for r in rows for k, v_ in r.items() if k != "flaw"}}]
                        for r in rows:
                            _append(path_out, {"config": config, "dataset": b.name, "seed": seed, "status": rec["status"],
                                               "column": v.column if v else None, "detail": v.detail if v else "", **r})
                    except Exception as e:  # noqa: BLE001
                        _append(path_out, {"config": config, "dataset": b.name, "seed": seed, "flaw": flaw,
                                           "status": "failed", "error": f"{type(e).__name__}: {e}"})
