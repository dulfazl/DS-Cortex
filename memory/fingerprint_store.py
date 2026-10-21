"""Adaptive memory (Pillar 3). Stores one row per completed run: the dataset fingerprint and the
chosen pipeline, never the raw data. Suggestions are advisory: the Architect may add them to its
candidates, the Critic may reject them, and nothing is ever applied silently."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app import db
from app.schemas import Recommendation
from tools.common import jsonable, now_iso
from tools.fingerprint_tools import fingerprint_distance

if TYPE_CHECKING:
    from workflows.pipeline_graph import PipelineRun

DISTANCE_THRESHOLD = 0.35   # above this, a bad analogy is worse than no suggestion
SEED_FILE = Path(__file__).resolve().parent / "seed_fingerprints.json"


def dataset_id(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(2_000_000)).hexdigest()[:16]


def save(dataset: str, task_family: str, vector: list[float], spec: dict, model: str, test_score: float,
         baseline_score: float, metric: str, ledger_ref: str, target_type: str | None, seeded: bool = False) -> None:
    with db.connect(write=True) as c:
        c.execute("INSERT INTO fingerprints(dataset_id, created_at, task_family, vector, pipeline_spec, model, test_score, "
                  "baseline_score, metric, ledger_ref, target_type, seeded) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (dataset, now_iso(), task_family, json.dumps(vector), json.dumps(spec), model, test_score,
                   baseline_score, metric, ledger_ref, target_type, int(seeded)))


def find_similar(fingerprint: list[float], task_family: str, k: int = 3, threshold: float = DISTANCE_THRESHOLD,
                 exclude_dataset: str | None = None) -> dict:
    """Same task family only; Euclidean distance on the normalized vector; nothing above the threshold."""
    with db.connect() as c:
        rows = c.execute("SELECT * FROM fingerprints WHERE task_family=?", (task_family,)).fetchall()
    scored = []
    for r in rows:
        if exclude_dataset and r["dataset_id"] == exclude_dataset:
            continue
        d = fingerprint_distance(fingerprint, json.loads(r["vector"]))
        scored.append({"id": r["id"], "dataset_id": r["dataset_id"], "created_at": r["created_at"],
                       "task_family": r["task_family"], "distance": d, "spec": json.loads(r["pipeline_spec"]),
                       "model": r["model"], "test_score": r["test_score"], "baseline_score": r["baseline_score"],
                       "metric": r["metric"], "target_type": r["target_type"]})
    scored.sort(key=lambda x: x["distance"])
    nearest = scored[0]["distance"] if scored else None
    matches = [s for s in scored if s["distance"] <= threshold][:k]
    return jsonable({"matches": matches, "nearest_distance": nearest, "threshold": threshold, "searched": len(scored)})


def lookup_for_run(run: PipelineRun) -> None:
    s = run.s
    prof = s.stats_profile
    exclude = dataset_id(s.file_path) if s.flags.memory_exclude_same else None
    res, ev = run.ctx.tool("memory.fingerprint_store.find_similar", find_similar,
                           {"task_family": s.task_plan.task_family, "k": 3, "threshold": DISTANCE_THRESHOLD},
                           prof.fingerprint, s.task_plan.task_family, 3, DISTANCE_THRESHOLD, exclude)
    seen, sugg = set(), []
    for m in res["matches"]:
        key = (m["spec"].get("model"), json.dumps(m["spec"].get("params"), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        sugg.append({**m, "spec": {k: m["spec"].get(k) for k in ("model", "params", "prep", "balance")}})
    s.memory = {"suggestions": sugg[:2], "lookup_run_id": ev.run_id, "nearest_distance": res["nearest_distance"],
                "searched": res["searched"]}
    if sugg:
        run.emit("memory_suggestion", {"suggestions": [{"model": m["model"], "distance": m["distance"], "date": m["created_at"],
                                                        "test_score": m["test_score"], "baseline": m["baseline_score"],
                                                        "metric": m["metric"]} for m in sugg[:2]]})
    run.ctx.ledger.add(Recommendation(agent="ml_architect", decision_point="memory_lookup",
                                      recommendation=(f"{len(sugg[:2])} suggestion(s) passed to the Architect and Critic as advice"
                                                      if sugg else "no similar past dataset close enough"),
                                      rationale=(f"nearest distance {res['nearest_distance']:.3f} against threshold "
                                                 f"{DISTANCE_THRESHOLD}" if res["nearest_distance"] is not None
                                                 else "the store has no runs of this task family yet"),
                                      evidence=[ev], confidence=1.0, confidence_method="rule"),
                       options=["use as advice", "ignore"], chosen="advice only, never auto-applied")


def log_memory_review(run: PipelineRun) -> None:
    mem = [c for c in run.s.candidates if c.origin == "memory"]
    run.s.memory["review"] = [{"id": c.id, "status": c.status, "objections": [o.message for o in c.objections]} for c in mem]


def record_outcome(run: PipelineRun) -> dict:
    """At the end of a completed run: did memory help? Then store this run's fingerprint."""
    s = run.s
    sugg = s.memory.get("suggestions") or []
    mem = [c for c in s.candidates if c.origin == "memory"]
    chosen = s.candidate(s.chosen_id)
    won = bool(chosen and (chosen.origin == "memory" or (chosen.parent or "").startswith("m")))
    from tools import pipeline_core as pc
    others = [c for c in s.candidates if c.origin != "memory" and c.cv and c.cv.primary_mean == c.cv.primary_mean]
    diff = None
    if mem and others:
        best_other = max(others, key=lambda c: pc.oriented(s.task_plan.metric, c.cv.primary_mean))
        best_mem = max(mem, key=lambda c: pc.oriented(s.task_plan.metric, c.cv.primary_mean))
        diff = pc.oriented(s.task_plan.metric, best_mem.cv.primary_mean) - pc.oriented(s.task_plan.metric, best_other.cv.primary_mean)
    row = {"suggested": len(sugg), "entered": len(mem), "won": int(won),
           "rejected": sum(1 for c in mem if c.status == "rejected"), "score_diff": diff,
           "distance": sugg[0]["distance"] if sugg else None}
    with db.connect(write=True) as c:
        c.execute("INSERT INTO memory_log(session_id, suggested, entered, won, rejected, score_diff, distance, data, created_at) "
                  "VALUES (?,?,?,?,?,?,?,?,?)", (s.id, row["suggested"], row["entered"], row["won"], row["rejected"],
                                                 row["score_diff"], row["distance"], json.dumps(row), now_iso()))
    test = s.results.get("test") or {}
    if chosen and test.get("score") is not None:
        save(dataset_id(s.file_path), s.task_plan.task_family, s.stats_profile.fingerprint,
             {k: chosen.spec.get(k) for k in ("model", "params", "prep", "balance")}, chosen.model, test["score"],
             test["baseline"]["score"], s.task_plan.metric, s.id, run._target_type())
    return row


def memory_stats() -> dict[str, Any]:
    with db.connect() as c:
        rows = [dict(r) for r in c.execute("SELECT suggested, entered, won, rejected, score_diff FROM memory_log").fetchall()]
        n = c.execute("SELECT COUNT(*) AS n FROM fingerprints").fetchone()["n"]
    return {"runs_logged": len(rows), "with_suggestion": sum(1 for r in rows if r["suggested"]),
            "entered": sum(r["entered"] for r in rows), "won": sum(r["won"] for r in rows),
            "rejected": sum(r["rejected"] for r in rows), "fingerprints_stored": n}


def export_seed(path: Path = SEED_FILE) -> int:
    """Write the stored fingerprints and pipelines (never raw data) to a JSON file for seeding."""
    with db.connect() as c:
        rows = [dict(r) for r in c.execute("SELECT dataset_id, task_family, vector, pipeline_spec, model, test_score, "
                                           "baseline_score, metric, target_type FROM fingerprints").fetchall()]
    path.write_text(json.dumps(rows, indent=1))
    return len(rows)


def load_seed(path: Path = SEED_FILE) -> int:
    """Fill an empty store from the committed seed file, so the memory demo survives a disk reset."""
    if not path.exists():
        return 0
    with db.connect() as c:
        if c.execute("SELECT COUNT(*) AS n FROM fingerprints").fetchone()["n"]:
            return 0
    rows = json.loads(path.read_text())
    for r in rows:
        save(r["dataset_id"], r["task_family"], json.loads(r["vector"]), json.loads(r["pipeline_spec"]), r["model"],
             r["test_score"], r["baseline_score"], r["metric"], "seed", r.get("target_type"), seeded=True)
    return len(rows)
