"""The conflict-resolution table, encoded as data and functions, not prompt text.
Each rule takes the debate state and returns a decision plus the rule name. The LLM may explain
a decision; it may not make it."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from agents.statistician.core import validity_issues
from app.schemas import CandidatePipeline, DebatePosition, TaskPlan
from tools import pipeline_core as pc

TIE_MARGIN = 0.10


@dataclass
class DebateState:
    positions: dict[str, DebatePosition]          # agent -> position (excluded ones removed)
    candidates: dict[str, CandidatePipeline]
    plan: TaskPlan
    explainability_waived: bool = False
    current: str | None = None


@dataclass
class Decision:
    chosen: str
    rule: str
    reason: str
    tiebreak: bool = False


@dataclass(frozen=True)
class Rule:
    name: str
    conflict: str
    default_resolution: str
    override_condition: str
    decide: Callable[[DebateState], Decision | None] = field(compare=False)
    blocking: bool = True


def _usable_claims(pos: DebatePosition, kind: str) -> list:
    return [c for c in pos.reasons if c.kind == kind and c.verified is not False]


def _validity(st: DebateState) -> Decision | None:
    a, s = st.positions.get("ml_architect"), st.positions.get("statistician")
    if not a or not s or a.preferred == s.preferred or st.current != a.preferred:
        return None
    if not validity_issues(st.candidates[a.preferred]) or not _usable_claims(s, "validity"):
        return None
    issues = ", ".join(sorted({o.check for o in validity_issues(st.candidates[a.preferred])}))
    return Decision(s.preferred, "validity_vs_performance",
                    f"{a.preferred} has the best raw score but fails {issues}; validity wins, so {s.preferred} is chosen")


def _explainability(st: DebateState) -> Decision | None:
    b = st.positions.get("business")
    if not b or not st.plan.explainability_required or st.explainability_waived or b.preferred == st.current:
        return None
    if validity_issues(st.candidates[b.preferred]) or not _usable_claims(b, "interpretability"):
        return None
    return Decision(b.preferred, "explainability_vs_metric",
                    f"the client asked for explainability; interpretable {b.preferred} is preferred over {st.current}")


def _architect_vs_business(st: DebateState) -> Decision | None:
    a, b = st.positions.get("ml_architect"), st.positions.get("business")
    explain = st.plan.explainability_required and not st.explainability_waived
    if not a or not b or explain or a.preferred == b.preferred or st.current != a.preferred:
        return None
    if validity_issues(st.candidates[b.preferred]):
        return Decision(a.preferred, "architect_vs_business",
                        f"business prefers {b.preferred}, but it has unresolved validity objections; "
                        f"{a.preferred} stands")
    gap = b.confidence - a.confidence
    if abs(gap) < TIE_MARGIN:
        ca, cb = st.candidates[a.preferred], st.candidates[b.preferred]
        metric = st.plan.metric
        win = ca if pc.oriented(metric, ca.cv.primary_mean) >= pc.oriented(metric, cb.cv.primary_mean) else cb
        return Decision(win.id, "architect_vs_business",
                        f"confidences {a.confidence:.2f} (architect) and {b.confidence:.2f} (business) are within "
                        f"{TIE_MARGIN:.0%}: supervisor tiebreak on cross-validated score picks {win.id}", tiebreak=True)
    win = b if gap > 0 else a
    return Decision(win.preferred, "architect_vs_business",
                    f"higher stated confidence wins: {win.agent} ({win.confidence:.2f} against "
                    f"{(a if win is b else b).confidence:.2f})")


def _research(st: DebateState) -> Decision | None:
    return None   # informational only: produces a report note, never a decision


RESOLUTION_TABLE: list[Rule] = [
    Rule("validity_vs_performance", "Statistical validity vs raw performance", "Validity wins",
         "Logged justification required", _validity),
    Rule("explainability_vs_metric", "Explainability need vs best raw metric", "Explainability wins if the client flagged it",
         "Client waives it explicitly", _explainability),
    Rule("architect_vs_business", "ML Architect vs Business Insight on model choice",
         "Higher stated confidence wins; flag if the gap is under 10 percent", "Supervisor tiebreak, logged with reasoning",
         _architect_vs_business),
    Rule("research_divergence", "Research agent flags divergence from established practice",
         "Non-blocking note in the report", "None; informational only", _research, blocking=False),
]


def resolve(state: DebateState) -> tuple[str, list[Decision]]:
    """Apply the table in order, starting from the Architect's pick."""
    arch = state.positions.get("ml_architect")
    state.current = arch.preferred if arch else next(iter(state.positions.values())).preferred
    applied: list[Decision] = []
    for rule in RESOLUTION_TABLE:
        if not rule.blocking:
            continue
        d = rule.decide(state)
        if d:
            applied.append(d)
            state.current = d.chosen
    return state.current, applied


def table_as_rows() -> list[dict]:
    return [{"rule": r.name, "conflict": r.conflict, "default_resolution": r.default_resolution,
             "override_condition": r.override_condition} for r in RESOLUTION_TABLE]
