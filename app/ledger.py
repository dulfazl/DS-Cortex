"""The Decision Ledger: nothing is decided silently. Each entry is written to SQLite and
to logs/ledger_<session>.jsonl."""
from __future__ import annotations

import json
from typing import Callable

from app import db
from app.config import get_settings
from app.schemas import DecidedBy, LedgerEntry, Recommendation, validate_recommendation
from tools.common import now_iso


class Ledger:
    def __init__(self, session_id: str, on_add: Callable[[LedgerEntry], None] | None = None):
        self.session_id = session_id
        self.on_add = on_add

    def _next_step(self) -> int:
        with db.connect() as c:
            row = c.execute("SELECT MAX(step) AS m FROM ledger WHERE session_id=?", (self.session_id,)).fetchone()
        return (row["m"] or 0) + 1

    def add(self, recommendation: Recommendation, options: list[str], chosen: str,
            decided_by: DecidedBy = "agent_rule", client_choice: str | None = None,
            overridden: bool = False, note: str | None = None, computed_available: bool = True) -> LedgerEntry:
        validate_recommendation(recommendation, computed_available=computed_available)
        entry = LedgerEntry(step=self._next_step(), decision_point=recommendation.decision_point,
                            options_considered=[str(o) for o in options], chosen=str(chosen),
                            recommendation=recommendation, decided_by=decided_by,
                            client_choice=client_choice, overridden=overridden,
                            timestamp=now_iso(), note=note)
        payload = entry.model_dump_json()
        with db.connect(write=True) as c:
            c.execute("INSERT INTO ledger(session_id, step, data, created_at) VALUES (?,?,?,?)",
                      (self.session_id, entry.step, payload, entry.timestamp))
        path = get_settings().log_dir / f"ledger_{self.session_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(payload + "\n")
        if self.on_add:
            self.on_add(entry)
        return entry

    def entries(self) -> list[LedgerEntry]:
        with db.connect() as c:
            rows = c.execute("SELECT data FROM ledger WHERE session_id=? ORDER BY step", (self.session_id,)).fetchall()
        return [LedgerEntry.model_validate_json(r["data"]) for r in rows]

    def export(self, fmt: str = "json") -> str:
        entries = self.entries()
        if fmt == "json":
            return json.dumps([e.model_dump() for e in entries], indent=2, default=str)
        if fmt in ("markdown", "md"):
            return ledger_markdown(entries)
        raise ValueError(f"Unknown ledger format: {fmt}")


def ledger_markdown(entries: list[LedgerEntry]) -> str:
    lines = ["# Decision Ledger", "",
             "| Step | Decision point | Chosen | Options | Confidence | Method | Decided by | Overridden |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in entries:
        r = e.recommendation
        opts = ", ".join(e.options_considered[:6]) + (" ..." if len(e.options_considered) > 6 else "")
        lines.append(f"| {e.step} | {e.decision_point} | {_md(e.chosen)} | {_md(opts)} | {r.confidence:.2f} | "
                     f"{r.confidence_method} | {e.decided_by} | {'yes' if e.overridden else 'no'} |")
    lines.append("")
    for e in entries:
        r = e.recommendation
        lines += [f"## Step {e.step}: {e.decision_point}", "",
                  f"- **Chosen:** {_md(e.chosen)}" + (f" (client chose {_md(e.client_choice)})" if e.client_choice else ""),
                  f"- **Agent:** {r.agent}",
                  f"- **Rationale:** {_md(r.rationale)}",
                  f"- **Evidence:** " + ", ".join(f"`{ev.tool}` run {ev.run_id[:8]}" for ev in r.evidence)]
        if e.note:
            lines.append(f"- **Note:** {_md(e.note)}")
        lines.append("")
    return "\n".join(lines)


def _md(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")
