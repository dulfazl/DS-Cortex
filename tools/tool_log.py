"""Every tool call goes through ToolLog.run: it executes the function, stores the real result
under a run_id, and returns an Evidence object. The verifier later checks claims against
these stored results, so the LLM cannot cite a run that never happened."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from typing import Any

from app import db
from app.schemas import Evidence
from tools.common import jsonable, now_iso


class ToolLog:
    def __init__(self, session_id: str, persist: bool = True,
                 on_run: Callable[[str, str, float], None] | None = None):
        self.session_id = session_id
        self.persist = persist
        self.on_run = on_run
        self._cache: dict[str, dict] = {}

    def run(self, tool: str, fn: Callable[..., Any], inputs: dict[str, Any] | None = None,
            *args, **kwargs) -> tuple[dict, Evidence]:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        if not isinstance(result, dict):
            result = {"value": result}
        result = jsonable(result)
        inputs = jsonable(inputs or {})
        run_id = uuid.uuid4().hex
        self.record(run_id, tool, inputs, result)
        if self.on_run:
            self.on_run(tool, run_id, time.perf_counter() - t0)
        return result, Evidence(tool=tool, inputs=inputs, result=result, run_id=run_id)

    def record(self, run_id: str, tool: str, inputs: dict, result: dict) -> None:
        self._cache[run_id] = {"run_id": run_id, "tool": tool, "inputs": inputs, "result": result}
        if self.persist:
            with db.connect(write=True) as c:
                c.execute("INSERT INTO tool_runs(run_id, session_id, tool, inputs, result, created_at) VALUES (?,?,?,?,?,?)",
                          (run_id, self.session_id, tool, json.dumps(inputs), json.dumps(result), now_iso()))

    def get(self, run_id: str) -> dict | None:
        if run_id in self._cache:
            return self._cache[run_id]
        if not self.persist:
            return None
        with db.connect() as c:
            row = c.execute("SELECT * FROM tool_runs WHERE run_id=? AND session_id=?", (run_id, self.session_id)).fetchone()
        if not row:
            return None
        rec = {"run_id": row["run_id"], "tool": row["tool"], "inputs": json.loads(row["inputs"]),
               "result": json.loads(row["result"])}
        self._cache[run_id] = rec
        return rec

    def all(self) -> list[dict]:
        if not self.persist:
            return list(self._cache.values())
        with db.connect() as c:
            rows = c.execute("SELECT run_id, tool, created_at FROM tool_runs WHERE session_id=? ORDER BY created_at",
                             (self.session_id,)).fetchall()
        return [dict(r) for r in rows]


def lookup(obj: Any, path: str) -> Any:
    """Follow a dotted path ("a.b.0.c") into nested dicts/lists. Raises KeyError if absent."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(path)
            cur = cur[part]
        elif isinstance(cur, list):
            cur = cur[int(part)]
        else:
            raise KeyError(path)
    return cur
