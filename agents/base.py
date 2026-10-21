"""Shared plumbing for agents: the context they run in (tool log, ledger, LLM, event emitter)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.ledger import Ledger
from app.llm import BaseLLM, OfflineLLM
from app.schemas import Evidence, Recommendation
from tools.tool_log import ToolLog


def _noop(event_type: str, data: dict) -> None:
    return None


@dataclass
class AgentContext:
    session_id: str
    log: ToolLog
    ledger: Ledger
    llm: BaseLLM = field(default_factory=OfflineLLM)
    emit: Callable[[str, dict], None] = _noop
    seed: int = 42
    fast: bool = False
    narration_depth: str = "technical"

    def tool(self, name: str, fn: Callable[..., Any], inputs: dict | None = None, *args, **kwargs) -> tuple[dict, Evidence]:
        return self.log.run(name, fn, inputs, *args, **kwargs)

    @classmethod
    def offline(cls, session_id: str = "test", persist: bool = False, **kw) -> "AgentContext":
        """A throwaway context for scripts and tests."""
        return cls(session_id=session_id, log=ToolLog(session_id, persist=persist), ledger=Ledger(session_id), **kw)


class BaseAgent:
    name = "agent"

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def started(self, detail: str = "") -> None:
        self.ctx.emit("agent_started", {"agent": self.name, "detail": detail})

    def finished(self, detail: str = "") -> None:
        self.ctx.emit("agent_finished", {"agent": self.name, "detail": detail})

    def record(self, decision_point: str, recommendation: str, rationale: str, evidence: list[Evidence],
               confidence: float, method: str, options: list[str], chosen: str | None = None,
               decided_by: str = "agent_rule", note: str | None = None, **kw) -> Recommendation:
        rec = Recommendation(agent=self.name, decision_point=decision_point, recommendation=recommendation,
                             rationale=rationale, evidence=evidence, confidence=round(float(confidence), 4),
                             confidence_method=method, **kw)
        self.ctx.ledger.add(rec, options=options, chosen=chosen or recommendation, decided_by=decided_by, note=note)
        return rec
