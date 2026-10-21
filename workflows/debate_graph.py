"""The debate protocol. It runs once, at the modelling bundle:
  1. Statistician, ML Architect, Business Insight and Research each state a position.
  2. The evidence verifier checks every claim against the tool log.
  3. The Supervisor finds genuine conflicts (positions that disagree).
  4. The resolution table decides.
  5. At most 2 rebuttal rounds, then a forced decision. Never loops."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from agents.base import AgentContext, evidence_for_run
from agents.business.core import BusinessInsightAgent
from agents.ml_architect.core import MLArchitectAgent
from agents.pipeline_critic.core import PipelineCriticAgent
from agents.research.core import ResearchAgent
from agents.statistician.core import StatisticianAgent, validity_issues
from app.llm import ask_json, load_prompt
from app.schemas import (CandidatePipeline, Check, Claim, DebatePosition, DebateTurn, Evidence,
                         StatisticalProfile, TaskPlan)
from tools.verify_tools import verify_claims
from workflows.resolution_table import DebateState, Decision, resolve

MAX_REBUTTAL_ROUNDS = 2


@dataclass
class DebateResult:
    chosen: str
    rule: str
    decisions: list[Decision]
    positions: list[DebatePosition]
    transcript: list[DebateTurn]
    verifier: dict
    notes: list[str] = field(default_factory=list)
    forced_tiebreak: bool = False
    consensus: bool = False


def _positions(ctx: AgentContext, survivors, plan, profile, round_no) -> tuple[dict[str, DebatePosition], list[str]]:
    pos = {
        "statistician": StatisticianAgent(ctx).position(survivors, plan, round_no),
        "ml_architect": MLArchitectAgent(ctx).position(survivors, plan, round_no),
        "business": BusinessInsightAgent(ctx).position(survivors, plan, round_no),
    }
    research, notes = ResearchAgent(ctx).position(survivors, plan, profile, round_no)
    pos["research"] = research
    return pos, notes


def _evidence_table(ctx: AgentContext, survivors: list[CandidatePipeline]) -> list[dict]:
    rows = []
    for c in survivors:
        rows.append({"candidate": c.id, "model": c.model, "run_id": c.eval_run_id,
                     "fields": {"primary_mean": round(c.cv.primary_mean, 6), "primary_std": round(c.cv.primary_std, 6),
                                "fit_time": round(c.cv.fit_time, 4)}})
        for o in c.objections:
            chk = o.claim.checks[0] if o.claim.checks else None
            if chk:
                rows.append({"candidate": c.id, "objection": o.check, "run_id": chk.run_id,
                             "fields": {chk.path: (o.claim.evidence[0].result if o.claim.evidence else {}).get(chk.path)}})
    return rows


def _llm_augment(ctx: AgentContext, pos: DebatePosition, survivors, plan: TaskPlan, verify: bool) -> DebatePosition:
    """The LLM may restate an agent's position and add up to two claims. Every claim must cite a
    run_id and a field path; the verifier decides whether it counts."""
    ids = [c.id for c in survivors]
    prompt = (f"{load_prompt(pos.agent)}\nYou are the {pos.agent} agent in a model-selection debate. Metric: {plan.metric}.\n"
              f"Current rule-based preference: {pos.preferred}.\nEvidence table (each row is a real tool run):\n"
              f"{json.dumps(_evidence_table(ctx, survivors))[:9000]}\n"
              "Reply as JSON {\"preferred\": <one of " + json.dumps(ids) + ">, \"claims\": [{\"text\": str, \"run_id\": str, "
              "\"path\": str, \"op\": \"approx\"|\"gt\"|\"lt\"|\"true\"|\"false\", \"value\": number|null}]} with at most two "
              "claims, each citing a run_id and field from the table.")
    out = ask_json(ctx.llm, prompt)
    if not isinstance(out, dict):
        return pos
    claims = []
    for c in (out.get("claims") or [])[:2]:
        if not isinstance(c, dict):
            continue
        run_id = str(c.get("run_id") or "")
        ev = evidence_for_run(ctx.log, run_id) or Evidence(tool="unknown", inputs={}, result={}, run_id=run_id or "missing")
        try:
            chk = Check(run_id=run_id or "missing", path=str(c.get("path") or ""), op=c.get("op") or "approx",
                        value=c.get("value"), tol=1e-3)
        except Exception:  # noqa: BLE001 - a malformed check simply fails verification
            chk = Check(run_id=run_id or "missing", path="__invalid__", op="true")
        claims.append(Claim(text=str(c.get("text", ""))[:300], kind="llm", source="llm", evidence=[ev], checks=[chk]))
    new = pos.model_copy(deep=True)
    new.reasons += claims
    pref = out.get("preferred")
    if pref in ids and pref != pos.preferred and claims:
        if not verify:
            new.preferred = pref
        else:
            verify_claims(claims, ctx.log)
            if any(c.verified for c in claims):
                new.preferred = pref
    return new


def run_debate(ctx: AgentContext, survivors: list[CandidatePipeline], plan: TaskPlan,
               profile: StatisticalProfile | None, df_train: pd.DataFrame, baseline_mean: float,
               verify: bool = True, llm_positions: bool = True, explainability_waived: bool = False) -> DebateResult:
    transcript: list[DebateTurn] = []
    notes: list[str] = []
    cands = {c.id: c for c in survivors}
    totals = {"total": 0, "verified": 0, "rejected": 0, "rejected_llm": 0, "total_llm": 0, "rejected_texts": []}
    decisions: list[Decision] = []
    chosen, consensus, forced = None, False, False

    if len(survivors) == 1:
        only = survivors[0].id
        return DebateResult(chosen=only, rule="single_survivor", decisions=[], positions=[], verifier=totals,
                            transcript=[DebateTurn(round=0, agent="supervisor", kind="decision",
                                                   text=f"only {only} survived the Critic; no debate needed")])

    for round_no in range(0, MAX_REBUTTAL_ROUNDS + 1):
        positions, rnotes = _positions(ctx, survivors, plan, profile, round_no)
        if round_no == 0:
            notes += rnotes
        if ctx.llm.available and llm_positions:
            positions = {k: _llm_augment(ctx, p, survivors, plan, verify) for k, p in positions.items()}
        for p in positions.values():
            transcript.append(DebateTurn(round=round_no, agent=p.agent, kind="position",
                                         text=f"prefers {p.preferred} (confidence {p.confidence:.2f}, {p.confidence_method})",
                                         claims=p.reasons))
        # --- verification
        all_claims = [c for p in positions.values() for c in p.reasons]
        if verify:
            counts = verify_claims(all_claims, ctx.log)
            for k in ("total", "verified", "rejected", "rejected_llm", "total_llm"):
                totals[k] += counts[k]
            totals["rejected_texts"] += counts["rejected_texts"]
            for p in positions.values():
                if p.reasons and not any(c.verified for c in p.reasons):
                    p.excluded = True
            transcript.append(DebateTurn(round=round_no, agent="verifier", kind="verification",
                                         text=f"{counts['verified']} of {counts['total']} claims verified; "
                                              f"{counts['rejected']} rejected and excluded from the decision",
                                         data={k: counts[k] for k in ("total", "verified", "rejected", "rejected_llm")}))
        else:
            totals["total"] += len(all_claims)
            totals["total_llm"] += sum(1 for c in all_claims if c.source == "llm")
        live = {k: p for k, p in positions.items() if not p.excluded}
        prefs = {p.preferred for p in live.values() if p.agent != "research"}
        if len(prefs) <= 1:
            chosen = next(iter(prefs)) if prefs else MLArchitectAgent.rank(survivors, plan.metric)[0].id
            consensus = True
            transcript.append(DebateTurn(round=round_no, agent="supervisor", kind="decision",
                                         text=f"all deciding agents agree on {chosen}"))
            break
        transcript.append(DebateTurn(round=round_no, agent="supervisor", kind="conflict",
                                     text="conflict: " + "; ".join(f"{p.agent} prefers {p.preferred}" for p in live.values())))
        state = DebateState(positions=live, candidates=cands, plan=plan, explainability_waived=explainability_waived)
        chosen, decisions = resolve(state)
        for d in decisions:
            transcript.append(DebateTurn(round=round_no, agent="supervisor", kind="resolution",
                                         text=f"[{d.rule}] {d.reason}", data={"rule": d.rule, "tiebreak": d.tiebreak}))
        # --- rebuttal: the Architect may answer a validity objection with new evidence (a re-run on a new seed)
        arch = live.get("ml_architect")
        lost_on_validity = bool(arch) and any(d.rule == "validity_vs_performance" for d in decisions)
        if not lost_on_validity:
            transcript.append(DebateTurn(round=round_no, agent="supervisor", kind="decision",
                                         text=f"no agent has new evidence to bring; the table decision stands: {chosen}"))
            break
        if round_no == MAX_REBUTTAL_ROUNDS:
            forced = True    # round cap reached without convergence: apply the table default, never loop
            transcript.append(DebateTurn(round=round_no, agent="supervisor", kind="decision",
                                         text=f"forced tiebreak: positions still disagree after {MAX_REBUTTAL_ROUNDS} "
                                              f"rebuttal rounds; the table default stands and {chosen} is chosen"))
            break
        top = cands[arch.preferred]
        checks = {o.check for o in validity_issues(top)}
        recheck = PipelineCriticAgent(ctx).review(top, df_train, plan, baseline_mean, only=checks,
                                                  seed_offset=97 * (round_no + 1))
        still = {o.check for o in recheck if o.severity in ("medium", "high", "blocker")}
        cleared = checks - still
        for o in top.objections:
            if o.check in cleared:
                o.resolved = True
        transcript.append(DebateTurn(round=round_no, agent="ml_architect", kind="rebuttal",
                                     text=(f"re-ran {', '.join(sorted(checks))} on {top.id} with a new seed: "
                                           + (f"{', '.join(sorted(cleared))} no longer flagged" if cleared else "the objection stands"))))
        # next round: positions are re-stated with the new evidence
    rule = decisions[-1].rule if decisions and not consensus else ("consensus" if consensus else "architect_default")
    return DebateResult(chosen=chosen, rule=rule, decisions=decisions, positions=list(positions.values()),
                        transcript=transcript, verifier=totals, notes=notes, forced_tiebreak=forced, consensus=consensus)
