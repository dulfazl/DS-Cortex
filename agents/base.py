"""Shared plumbing for agents: the context they run in (tool log, ledger, LLM, event emitter)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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
    def offline(cls, session_id: str = "test", persist: bool = False, **kw) -> AgentContext:
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


def slice_evidence(ev: Evidence, key: str | None, value: Any = None) -> Evidence:
    """Evidence for one column/item of a larger tool result. Same run_id, so the verifier still
    checks it against the full logged result."""
    if key is None:
        return ev
    return Evidence(tool=ev.tool, inputs={**ev.inputs, "item": key}, result={key: value}, run_id=ev.run_id)


def evidence_for_run(log, run_id: str | None) -> Evidence | None:
    """Rebuild an Evidence object from the tool log (the logged result, not a copy)."""
    rec = log.get(run_id) if run_id else None
    if rec is None:
        return None
    return Evidence(tool=rec["tool"], inputs=rec["inputs"], result=rec["result"], run_id=run_id)


def cv_claim(log, cand, text: str | None = None):
    """A claim about a candidate's CV score, checkable against its evaluation run."""
    from app.schemas import Check, Claim
    ev = evidence_for_run(log, cand.eval_run_id)
    if ev is None:
        return None
    return Claim(kind="cv", text=text or f"{cand.id} scores {cand.cv.primary_mean:.4f} {cand.cv.primary} in cross-validation "
                 f"(sd {cand.cv.primary_std:.4f})",
                 evidence=[ev], checks=[Check(run_id=ev.run_id, path="primary_mean", op="approx",
                                              value=cand.cv.primary_mean, tol=1e-4)])
