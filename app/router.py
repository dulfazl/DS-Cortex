"""HTTP API. Uploads are size-limited, CSV-only, stored under a random folder and never executed."""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agents.router.core import RouterAgent
from app import events
from app.chat import answer as chat_answer
from app.config import get_settings
from app.ledger import Ledger
from app.llm import get_llm
from app.runner import get_runner
from app.session import RunFlags, Session, store
from tools import drift_check_tools as dc
from tools import pipeline_core as pc
from tools.common import DataError, jsonable
from tools.pandas_tools import coerce_types, load_csv, quick_profile
from tools.tool_log import ToolLog

api = APIRouter()
TERMINAL = {"completed", "failed"}
EXPORT_FILES = {"report.md", "report.html", "report.pdf", "pipeline.py", "requirements.txt", "ledger.json", "ledger.md",
                "debate_transcript.md", "manifest.json", "model/model_card.md", "model/input_schema.json"}
_rate: dict[str, deque] = defaultdict(deque)
RATE_LIMIT = (20, 3600)   # session creations per IP per hour


# --------------------------------------------------------------------------- helpers
def _session(sid: str) -> Session:
    if not re.fullmatch(r"[0-9a-f]{32}", sid or ""):
        raise HTTPException(404, "session not found")
    s = store.get(sid)
    if s is None:
        raise HTTPException(404, "session not found")
    return s


def _check_rate(ip: str) -> None:
    n, window = RATE_LIMIT
    q, now = _rate[ip], time.time()
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= n:
        raise HTTPException(429, f"too many new sessions from this address; try again in {int(window - (now - q[0]))} seconds")
    q.append(now)


def _safe_name(name: str) -> str:
    base = Path(name or "upload.csv").name
    base = re.sub(r"[^\w.\-]", "_", base)[:80]
    return base if base.lower().endswith(".csv") else base + ".csv"


async def _save_upload(file: UploadFile, folder: Path) -> Path:
    settings = get_settings()
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "only .csv files are accepted")
    limit = settings.max_upload_mb * 1024 * 1024
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / _safe_name(file.filename)
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                out.close()
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(413, f"file is larger than {settings.max_upload_mb} MB")
            out.write(chunk)
    if size == 0:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, "the file is empty")
    return dest


def summary(s: Session) -> dict:
    r = get_runner()
    return jsonable({"id": s.id, "status": s.status, "mode": s.mode, "file_name": s.file_name,
                     "problem_statement": s.problem_statement, "created_at": s.created_at, "updated_at": s.updated_at,
                     "profile": s.profile, "task_plan": s.task_plan.model_dump() if s.task_plan else None,
                     "pending": s.pending, "current_stage": s.current_stage, "completed_stages": s.completed_stages,
                     "chosen_id": s.chosen_id, "error": s.error, "running": r.is_active(s.id),
                     "queue_position": r.queue_position(s.id), "flags": s.flags.model_dump(), "usage": s.usage,
                     "memory": {k: v for k, v in s.memory.items() if k in ("suggestions", "nearest_distance")}})


# --------------------------------------------------------------------------- endpoints
@api.post("/sessions")
async def create_session(request: Request, file: UploadFile = File(...), problem_statement: str = Form(...),
                         mode: str = Form("autopilot")):
    _check_rate(request.client.host if request.client else "unknown")
    if mode not in ("autopilot", "review"):
        raise HTTPException(422, "mode must be autopilot or review")
    statement = problem_statement.strip()[:2000]
    if not statement:
        raise HTTPException(422, "describe the problem in a sentence")
    sid = uuid.uuid4().hex
    path = await _save_upload(file, get_settings().upload_dir / sid)
    try:
        profile = quick_profile(path, get_settings().max_profile_rows)
    except DataError as e:
        shutil.rmtree(path.parent, ignore_errors=True)
        raise HTTPException(422, str(e)) from e
    s = store.create(str(path), statement, mode=mode, file_name=path.name, session_id=sid)
    s.profile = profile
    store.save(s)
    events.emit(sid, "status", {"status": "created"})
    return {"session_id": sid, "profile": profile}


class StartBody(BaseModel):
    mode: str | None = None
    flags: dict[str, bool] | None = None


@api.post("/sessions/{sid}/start")
def start(sid: str, body: StartBody | None = None):
    s = _session(sid)
    body = body or StartBody()
    if body.mode in ("autopilot", "review"):
        s.mode = body.mode
    if body.flags:
        s.flags = RunFlags(**{**s.flags.model_dump(), **{k: v for k, v in body.flags.items() if k in RunFlags.model_fields}})
    if s.status in ("completed",):
        raise HTTPException(409, "this run is complete; use rerun to start a new one")
    if s.status in ("awaiting_approval", "awaiting_clarification", "blocked"):
        raise HTTPException(409, "the run is waiting for your answer")
    store.save(s)
    started = get_runner().start(sid)
    return {"started": started, "status": store.get(sid).status}


@api.get("/sessions/{sid}")
def get_session(sid: str):
    return summary(_session(sid))


@api.get("/sessions/{sid}/events")
async def stream(sid: str, request: Request, after: int = 0):
    _session(sid)
    last = request.headers.get("last-event-id")
    start_after = int(last) if last and last.isdigit() else after

    async def gen():
        seq = start_after
        idle = 0
        while True:
            if await request.is_disconnected():
                break
            batch = events.events_after(sid, seq)
            for ev in batch:
                seq = ev["seq"]
                yield {"event": ev["type"], "id": str(ev["seq"]), "data": json.dumps({**ev["data"], "seq": ev["seq"], "ts": ev["ts"]})}
            s = store.get(sid)
            if not batch and s and s.status in TERMINAL and not get_runner().is_active(sid):
                idle += 1
                if idle > 2:
                    break
            await asyncio.sleep(0.4)

    return EventSourceResponse(gen(), ping=15, headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class ClarifyBody(BaseModel):
    choice: str


@api.post("/sessions/{sid}/clarify")
def clarify(sid: str, body: ClarifyBody):
    s = _session(sid)
    if not s.pending or s.pending.get("kind") != "clarification":
        raise HTTPException(409, "no clarifying question is pending")
    cols = [c["name"] for c in s.profile["column_info"]]
    ans = RouterAgent.parse_answer(s.pending["payload"], body.choice, cols)
    if "free_text" in ans:
        raise HTTPException(422, f"'{body.choice}' is not a column or an option; pick one of the options")
    s.answers["clarify"] = ans
    s.pending, s.status = None, "running"
    store.save(s)
    get_runner().start(sid)
    return {"accepted": ans}


class ApproveBody(BaseModel):
    bundle: str | None = None
    decision: str = "approve"            # approve | change
    disabled: list[Any] | None = None    # preprocessing: actions to switch off
    options: dict[str, bool] | None = None   # feature engineering toggles
    choice: str | None = None            # modelling: the pipeline the client wants instead
    justification: str | None = None
    blockers: dict[str, dict] | None = None   # {column: {decision: drop|keep, explanation}}


@api.post("/sessions/{sid}/approve")
def approve(sid: str, body: ApproveBody):
    s = _session(sid)
    p = s.pending
    if not p or p.get("kind") not in ("approval", "blocker"):
        raise HTTPException(409, "nothing is waiting for approval")
    warning = None
    if p["kind"] == "blocker":
        ans = body.blockers or {c: {"decision": "drop"} for c in p["payload"]["blockers"]}
        for col, a in ans.items():
            if a.get("decision") == "keep" and not (a.get("explanation") or "").strip():
                raise HTTPException(422, f"explain why '{col}' is available at prediction time, or drop it")
        s.answers["blockers"] = ans
    else:
        ans = body.model_dump(exclude_none=True)
        if p["key"] == "modelling" and body.decision == "change":
            if not body.choice or s.candidate(body.choice) is None:
                raise HTTPException(422, "choose one of the candidate pipelines")
            if s.debate and s.debate.rule == "validity_vs_performance" and not (body.justification or "").strip():
                raise HTTPException(422, "the recommendation was decided on statistical validity: a justification is required to override it")
            from workflows.pipeline_graph import override_warning
            if f"override:{body.choice}" not in s.warnings_given:
                warning = override_warning(s, body.choice)
        s.answers[p["key"]] = ans
    s.pending, s.status = None, "running"
    store.save(s)
    get_runner().start(sid)
    return {"accepted": True, "warning": warning}


@api.get("/sessions/{sid}/results")
def results(sid: str):
    s = _session(sid)
    if s.status != "completed":
        raise HTTPException(409, f"results are not ready (status: {s.status})")
    from workflows.report_graph import load_explain
    explain = load_explain(s)
    entries = Ledger(s.id).entries()
    return jsonable({
        "session": summary(s), "plan": s.task_plan.model_dump(), "audit": s.audit_report.model_dump(),
        "stats": {"flags": [f.model_dump(exclude={"evidence"}) for f in s.stats_profile.flags],
                  "guidance": s.stats_profile.guidance, "fingerprint": s.stats_profile.fingerprint,
                  "fingerprint_features": s.stats_profile.fingerprint_features,
                  "top_correlations": s.stats_profile.top_correlations},
        "candidates": [c.model_dump() for c in s.candidates], "debate": s.debate.model_dump() if s.debate else None,
        "final": s.results.get("final_recommendation"), "test": s.results.get("test"),
        "calibration": s.results.get("calibration"), "override": s.results.get("override"),
        "explain": {k: explain.get(k) for k in ("importance", "learning_curve", "narrative", "narrative_beginner",
                                                 "business", "research", "cluster_profiles")},
        "errors": {k: v for k, v in (explain.get("errors") or {}).items() if k in
                   ("confusion_matrix", "classes", "per_class", "residual_mean", "residual_std", "worst", "sizes",
                    "threshold", "flagged", "top_rows", "positive_class")},
        "plots": explain.get("plots", []), "ledger": [e.model_dump() for e in entries], "memory": s.memory,
        "export": {"files": s.results.get("export", {}).get("files", [])}, "clean": s.stage_outputs.get("clean"),
        "baseline_cv": s.stage_outputs.get("baseline"), "usage": s.usage,
    })


@api.get("/sessions/{sid}/ledger")
def ledger(sid: str):
    """The ledger so far (also for failed or paused runs: the partial ledger is kept)."""
    _session(sid)
    return [e.model_dump() for e in Ledger(sid).entries()]


@api.get("/sessions/{sid}/export")
def export(sid: str, file: str | None = None):
    s = _session(sid)
    exp = s.results.get("export")
    if not exp:
        raise HTTPException(409, "the export is not ready")
    if file is None:
        return FileResponse(exp["zip"], filename=Path(exp["zip"]).name, media_type="application/zip")
    if file not in EXPORT_FILES:
        raise HTTPException(404, "unknown export file")
    path = Path(exp["dir"]) / file
    if not path.exists():
        raise HTTPException(404, f"{file} was not generated for this run")
    return FileResponse(path, filename=Path(file).name)


class ChatBody(BaseModel):
    question: str


@api.post("/sessions/{sid}/chat")
def chat(sid: str, body: ChatBody):
    s = _session(sid)
    q = body.question.strip()[:500]
    if not q:
        raise HTTPException(422, "ask a question")
    return chat_answer(s, q, get_llm())


def _new_batch(s: Session, path: Path) -> pd.DataFrame:
    df = coerce_types(load_csv(path))
    clean, _ = pc.apply_cleaning_plan(df, [a.model_dump() for a in s.audit_report.cleaning_plan
                                            if a.action not in ("drop_duplicates", "drop_rows_missing_target")])
    return clean


@api.post("/sessions/{sid}/drift")
async def drift(sid: str, file: UploadFile = File(...)):
    s = _session(sid)
    if s.status != "completed":
        raise HTTPException(409, "run the analysis first; the drift check compares against its training data")
    path = await _save_upload(file, s.dir / "batches" / uuid.uuid4().hex)
    try:
        new = _new_batch(s, path)
    except DataError as e:
        raise HTTPException(422, str(e)) from e
    ref_all = pd.read_pickle(s.dir / "clean.pkl")
    import numpy as np
    tr = np.load(s.dir / "split.npz")["train"]
    ref = ref_all.iloc[tr]
    exclude = [c for c in (s.task_plan.target, s.task_plan.time_column) if c]
    res, ev = ToolLog(sid).run("drift_check_tools.drift_check", dc.drift_check, {"batch": path.name, "rows": len(new)},
                               ref, new, exclude)
    events.emit(sid, "status", {"status": "drift_checked", "drift_score": res["drift_score"], "alert": res["alert"]})
    return {**res, "run_id": ev.run_id}


@api.post("/sessions/{sid}/predict")
async def predict(sid: str, file: UploadFile = File(...)):
    s = _session(sid)
    if s.status != "completed" or not s.results.get("model_file"):
        raise HTTPException(409, "no fitted model is available for this run")
    if s.task_plan.task_family not in ("classification", "regression"):
        raise HTTPException(400, "prediction export is available for classification and regression")
    path = await _save_upload(file, s.dir / "batches" / uuid.uuid4().hex)
    new = _new_batch(s, path)
    schema = json.loads((s.dir / "export" / "model" / "input_schema.json").read_text())
    missing = [c for c in schema["columns"] if c not in new.columns]
    if missing:
        raise HTTPException(422, f"the batch is missing required columns: {', '.join(missing)}")
    import joblib
    model = joblib.load(s.results["model_file"]["path"])
    X = new[list(schema["columns"])]
    pred = model.predict(X)
    out: dict[str, Any] = {"rows": len(X)}
    if s.task_plan.task_family == "classification":
        classes = s.results["predictions"]["classes"]
        out["predictions"] = [classes[int(i)] for i in pred]
        out["probabilities"] = model.predict_proba(X).round(4).tolist()
    else:
        out["predictions"] = [float(v) for v in pred]
    ref = pd.read_pickle(s.dir / "clean.pkl")
    out["drift"] = dc.drift_check(ref, new, [c for c in (s.task_plan.target, s.task_plan.time_column) if c])
    return jsonable(out)


@api.post("/sessions/{sid}/rerun")
def rerun(sid: str):
    s = _session(sid)
    new = store.create(s.file_path, s.problem_statement, mode=s.mode, file_name=s.file_name, flags=s.flags)
    new.profile = s.profile
    store.save(new)
    get_runner().start(new.id)
    return {"session_id": new.id}


@api.get("/sessions")
def list_sessions(limit: int = 20):
    return store.list()[:limit]


@api.get("/meta")
def meta():
    from tools.router_tools import METRICS
    from workflows.resolution_table import table_as_rows
    s = get_settings()
    llm = get_llm()
    return {"llm": llm.name, "llm_available": llm.available, "max_upload_mb": s.max_upload_mb,
            "metrics": METRICS, "resolution_table": table_as_rows(), "drift_limits": dc.LIMITS,
            "privacy": ("Only profiles and tool outputs are sent to the language model, but column names and some "
                        "category values go with them. On the free Gemini tier Google may use submitted content to "
                        "improve its products: do not upload confidential or personal data.")}
