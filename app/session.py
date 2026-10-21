"""Session state, persisted to SQLite after every change so a crashed run can resume."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app import db
from app.config import get_settings
from app.schemas import (CandidatePipeline, DataAuditReport, DebateOutcome, StatisticalProfile,
                         TaskPlan)
from tools.common import now_iso

RunMode = Literal["autopilot", "review"]
Status = Literal["created", "queued", "running", "awaiting_approval", "awaiting_clarification",
                 "blocked", "completed", "failed", "interrupted"]


class RunFlags(BaseModel):
    """Which pillars are switched on. The eval ablation flips these."""
    critic: bool = True
    debate: bool = True
    verify: bool = True
    memory: bool = True
    tuning: bool = True
    explain: bool = True
    llm_positions: bool = True
    memory_exclude_same: bool = False   # eval: suggestions only from other datasets
    force_shap_failure: bool = False    # tests: exercise the permutation-importance fallback


class Session(BaseModel):
    id: str
    created_at: str
    updated_at: str
    file_path: str
    file_name: str
    problem_statement: str
    mode: RunMode = "autopilot"
    status: Status = "created"
    flags: RunFlags = Field(default_factory=RunFlags)
    seed: int = 42
    profile: dict[str, Any] | None = None
    task_plan: TaskPlan | None = None
    audit_report: DataAuditReport | None = None
    stats_profile: StatisticalProfile | None = None
    candidates: list[CandidatePipeline] = Field(default_factory=list)
    debate: DebateOutcome | None = None
    chosen_id: str | None = None
    pending: dict[str, Any] | None = None          # {kind, key, payload}
    answers: dict[str, Any] = Field(default_factory=dict)
    completed_stages: list[str] = Field(default_factory=list)
    current_stage: str | None = None
    stage_outputs: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    warnings_given: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @property
    def dir(self) -> Path:
        d = get_settings().sessions_dir / self.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def candidate(self, cid: str) -> CandidatePipeline | None:
        return next((c for c in self.candidates if c.id == cid), None)


class SessionStore:
    def create(self, file_path: str, statement: str, mode: RunMode = "autopilot", file_name: str | None = None,
               flags: RunFlags | None = None, seed: int | None = None, session_id: str | None = None) -> Session:
        sid = session_id or uuid.uuid4().hex
        ts = now_iso()
        s = Session(id=sid, created_at=ts, updated_at=ts, file_path=str(file_path),
                    file_name=file_name or Path(file_path).name, problem_statement=statement, mode=mode,
                    flags=flags or RunFlags(), seed=seed if seed is not None else get_settings().seed)
        self.save(s)
        return s

    def save(self, s: Session) -> Session:
        s.updated_at = now_iso()
        with db.connect(write=True) as c:
            c.execute(
                "INSERT INTO sessions(id, data, status, created_at, updated_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET data=excluded.data, status=excluded.status, updated_at=excluded.updated_at",
                (s.id, s.model_dump_json(), s.status, s.created_at, s.updated_at))
        return s

    def get(self, sid: str) -> Session | None:
        with db.connect() as c:
            row = c.execute("SELECT data FROM sessions WHERE id=?", (sid,)).fetchone()
        return Session.model_validate_json(row["data"]) if row else None

    def update(self, sid: str, **fields) -> Session:
        s = self.get(sid)
        if s is None:
            raise KeyError(sid)
        for k, v in fields.items():
            setattr(s, k, v)
        return self.save(s)

    def list(self, status: str | None = None) -> list[dict]:
        q = "SELECT id, status, created_at, updated_at FROM sessions"
        args: tuple = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        with db.connect() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY created_at DESC", args).fetchall()]

    def delete(self, sid: str) -> None:
        with db.connect(write=True) as c:
            for t in ("sessions", "ledger", "tool_runs", "events"):
                c.execute(f"DELETE FROM {t} WHERE {'id' if t == 'sessions' else 'session_id'}=?", (sid,))
        shutil.rmtree(get_settings().sessions_dir / sid, ignore_errors=True)


store = SessionStore()
