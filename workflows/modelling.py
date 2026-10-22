"""The Architect-Critic revision loop: the Critic attacks, the Architect revises and resubmits,
the Critic re-checks. Capped at 3 rounds; every round is logged."""
from __future__ import annotations

import pandas as pd

from agents.base import AgentContext
from agents.ml_architect.core import MLArchitectAgent
from agents.pipeline_critic.core import PipelineCriticAgent
from app.schemas import CandidatePipeline, Recommendation, TaskPlan
from tools import pipeline_core as pc

MAX_ROUNDS = 3
REVISABLE = {"overfit_check", "shift_sensitivity", "seed_stability", "unnecessary_step_check"}
REJECTING = {"blocker"}


def _severe(objs) -> list:
    return [o for o in objs if not o.resolved and o.severity in ("medium", "high", "blocker")]


def critic_loop(ctx: AgentContext, architect: MLArchitectAgent, critic: PipelineCriticAgent,
                cands: list[CandidatePipeline], df_train: pd.DataFrame, plan: TaskPlan, baseline_mean: float,
                top_feature: str | None = None, top_k: int = 4, max_rounds: int = MAX_ROUNDS,
                memory_suggestions: dict[str, dict] | None = None, target_type: str | None = None) -> list[CandidatePipeline]:
    """Returns all candidates with statuses set; survivors have status 'active'."""
    critic.started(f"reviewing the top {top_k} candidates, up to {max_rounds} revision round{'s' if max_rounds != 1 else ''}")
    ranked = MLArchitectAgent.rank(cands, plan.metric)
    reviewed = ranked[:top_k]
    for c in ranked[top_k:]:
        c.status = "rejected"
        c.rationale += f" | not reviewed: outside the top {top_k} by cross-validated {plan.metric}"
    keep = {c.id for c in ranked[top_k:] if c.origin == "memory"}   # memory suggestions are always reviewed
    if plan.explainability_required and not any(c.interpretable for c in reviewed):
        best_interp = next((c for c in ranked[top_k:] if c.interpretable), None)
        if best_interp:
            keep.add(best_interp.id)   # the client asked for explainability: an interpretable option must reach the debate
    for c in ranked[top_k:]:
        if c.id in keep:
            c.status = "active"
            c.rationale = c.rationale.split(" | not reviewed")[0]
            reviewed.append(c)
    for c in reviewed:
        c.objections = critic.review(c, df_train, plan, baseline_mean, top_feature)
        if c.origin == "memory" and memory_suggestions and c.id in memory_suggestions:
            o = critic.review_memory_suggestion(c, memory_suggestions[c.id], plan, target_type, baseline_mean)
            if o:
                c.objections.append(o)
    # a memory suggestion whose analogy does not hold is rejected now, before any revision can launder it
    for c in reviewed:
        bad = [o for o in c.objections if o.check == "memory_fit_check"]
        if bad:
            c.status = "rejected"
            c.rationale += f" | rejected by the Critic: {bad[0].message}"
            ctx.ledger.add(Recommendation(agent="pipeline_critic", decision_point=f"reject:{c.id}",
                                          recommendation=f"reject the memory suggestion {c.id}", rationale=bad[0].message,
                                          evidence=bad[0].claim.evidence, confidence=0.95, confidence_method="rule"),
                           options=["keep", "reject"], chosen="reject")
    active = [c for c in reviewed if c.status != "rejected"]
    stuck: set[str] = set()       # candidates where a revision did not help: not revised again
    for round_no in range(1, max_rounds + 1):
        to_revise = [c for c in active if c.status == "active" and c.id not in stuck and
                     any(o.check in REVISABLE and not o.resolved and o.severity != "low" or
                         (o.check == "unnecessary_step_check" and not o.resolved) for o in c.objections)]
        if not to_revise:
            break
        changes = []
        for c in to_revise:
            objs = [o for o in c.objections if o.check in REVISABLE and not o.resolved]
            new = architect.revise(c, objs, df_train, plan, round_no)
            if new is None:
                changes.append(f"{c.id}: no revision helps")
                stuck.add(c.id)
                continue
            new.objections = critic.review(new, df_train, plan, baseline_mean, top_feature,
                                           only={o.check for o in objs} | {"baseline_check"})
            old_bad, new_bad = len(_severe(c.objections)), len(_severe(new.objections))
            # accept only if it fixes something without a real loss of score (within one CV standard deviation),
            # or keeps the same objections and scores better
            score_ok = pc.oriented(plan.metric, new.cv.primary_mean) >= pc.oriented(plan.metric, c.cv.primary_mean) - c.cv.primary_std
            better = architect.rank([c, new], plan.metric)[0].id == new.id
            if (new_bad < old_bad and score_ok) or (new_bad == old_bad and better):
                c.status = "rejected"
                c.rationale += f" | superseded by {new.id} in round {round_no}"
                active[active.index(c)] = new
                cands.append(new)
                changes.append(f"{c.id} -> {new.id} ({new.rationale.split(': ', 1)[-1]})")
            else:
                for o in objs:
                    o.message += " (still flagged after revision)"
                changes.append(f"{c.id}: revision {new.id} did not help; kept the original")
                stuck.add(c.id)
                new.status = "rejected"
                new.rationale += " | revision did not resolve the objections"
                cands.append(new)
        evidence = [o.claim.evidence[0] for c in to_revise for o in c.objections if o.claim.evidence][:8]
        if evidence:
            ctx.ledger.add(Recommendation(agent="pipeline_critic", decision_point=f"critic_round_{round_no}",
                                          recommendation="; ".join(changes), rationale=f"{len(to_revise)} candidate(s) revised",
                                          evidence=evidence, confidence=1.0, confidence_method="rule"),
                           options=[c.id for c in to_revise], chosen="; ".join(changes))
        ctx.emit("critic_round", {"round": round_no, "changes": changes})
    for c in active:
        if any(o.severity in REJECTING and not o.resolved for o in c.objections):
            c.status = "rejected"
            why = "; ".join(o.message for o in c.objections if o.severity in REJECTING)
            c.rationale += f" | rejected by the Critic: {why}"
            ev = [o.claim.evidence[0] for o in c.objections if o.claim.evidence][:3]
            ctx.ledger.add(Recommendation(agent="pipeline_critic", decision_point=f"reject:{c.id}",
                                          recommendation=f"reject {c.id}", rationale=why, evidence=ev,
                                          confidence=0.95, confidence_method="rule"),
                           options=["keep", "reject"], chosen="reject")
    survivors = [c for c in active if c.status == "active"]
    if not survivors and active:      # never leave the debate empty: keep the best, objections on record
        best = MLArchitectAgent.rank(active, plan.metric)[0]
        best.status = "active"
        best.rationale += " | kept although every candidate drew objections"
    critic.finished(f"{len([c for c in active if c.status == 'active'])} survivors")
    return cands
