"""Per-session event log with sequence numbers, stored in SQLite so a client can resume after a
dropped connection and a restarted server still has the history."""
from __future__ import annotations

import json
import threading
from typing import Any

from app import db
from tools.common import jsonable, now_iso

EVENT_TYPES = {"agent_started", "agent_finished", "tool_called", "ledger_entry", "plot_ready", "approval_required",
               "clarification_required", "blocker", "completed", "failed", "status", "queued", "stage",
               "candidate_evaluated", "critic_review", "critic_round", "memory_suggestion", "warning"}
_lock = threading.Lock()


def emit(session_id: str, event_type: str, data: dict[str, Any] | None = None) -> int:
    payload = json.dumps(jsonable(data or {}), default=str)
    with _lock, db.connect(write=True) as c:
        row = c.execute("SELECT COALESCE(MAX(seq), 0) AS m FROM events WHERE session_id=?", (session_id,)).fetchone()
        seq = row["m"] + 1
        c.execute("INSERT INTO events(session_id, seq, type, data, created_at) VALUES (?,?,?,?,?)",
                  (session_id, seq, event_type, payload, now_iso()))
    return seq


def events_after(session_id: str, after: int = 0, limit: int = 500) -> list[dict]:
    with db.connect() as c:
        rows = c.execute("SELECT seq, type, data, created_at FROM events WHERE session_id=? AND seq>? ORDER BY seq LIMIT ?",
                         (session_id, after, limit)).fetchall()
    return [{"seq": r["seq"], "type": r["type"], "data": json.loads(r["data"] or "{}"), "ts": r["created_at"]} for r in rows]
