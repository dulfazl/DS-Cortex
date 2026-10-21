"""Supervisor: runs the debate graph, applies the resolution table, writes the final
Recommendation (chosen pipeline, computed confidence, runner-up, transcript) to the ledger."""
from __future__ import annotations

import pandas as pd

from agents.base import BaseAgent
from agents.ml_architect.core import MLArchitectAgent
from app.llm import narrate
from app.schemas import (CandidatePipeline, DebateOutcome, DebateTurn, Recommendation, StatisticalProfile,
                         TaskPlan)
from tools import sklearn_tools as sk
from workflows.debate_graph import run_debate


class SupervisorAgent(BaseAgent):
    name = "supervisor"

    def decide(self, survivors: list[CandidatePipeline], plan: TaskPlan, profile: StatisticalProfile | None,
               df_train: pd.DataFrame, baseline_mean: float, verify: bool = True, llm_positions: bool = True,
               debate: bool = True) -> DebateOutcome:
        self.started("running the debate" if debate else "picking the best cross-validated candidate")
        cands = {c.id: c for c in survivors}
        if debate:
            res = run_debate(self.ctx, survivors, plan, profile, df_train, baseline_mean, verify=verify,
                             llm_positions=llm_positions)
            chosen_id, rule, transcript, positions = res.chosen, res.rule, res.transcript, res.positions
            verifier, notes, forced = res.verifier, res.notes, res.forced_tiebreak
            reasons = [d.reason for d in res.decisions]
        else:   # sequential pipeline (ablation config C): best CV score, no critique, no debate
            chosen_id = MLArchitectAgent.rank(survivors, plan.metric)[0].id
            rule, transcript, positions, verifier, notes, forced = "best_cv_no_debate", [], [], {"total": 0, "rejected": 0}, [], False
            reasons = [f"{chosen_id} has the best cross-validated {plan.metric}"]
        chosen = cands[chosen_id]
        others = [c for c in MLArchitectAgent.rank(survivors, plan.metric) if c.id != chosen_id]
        runner = others[0] if others else None
        conf, ev = self.ctx.tool("sklearn_tools.evidence_confidence", sk.evidence_confidence,
                                 {"chosen": chosen_id, "runner_up": runner.id if runner else None},
                                 chosen.cv.model_dump(), runner.cv.model_dump() if runner else None, baseline_mean,
                                 plan.metric, self.ctx.seed)
        winning = [c for p in positions if p.preferred == chosen_id and not p.excluded for c in p.reasons
                   if c.verified is not False]
        evidence, seen = [ev], {ev.run_id}
        for c in winning:
            for e in c.evidence:
                if e.run_id not in seen:
                    evidence.append(e)
                    seen.add(e.run_id)
        facts = {"chosen": chosen_id, "runner_up": runner.id if runner else None, "metric": plan.metric,
                 "chosen_cv_mean": chosen.cv.primary_mean, "chosen_cv_std": chosen.cv.primary_std,
                 "runner_cv_mean": runner.cv.primary_mean if runner else None, "baseline_mean": baseline_mean,
                 "confidence": conf["confidence"], "rule": rule, "reasons": reasons}
        fallback = (f"{chosen_id} chosen by rule '{rule}': " + " ".join(reasons) +
                    f" Confidence {conf['confidence']:.2f} from CV stability, bootstrap separation from "
                    f"{runner.id if runner else 'no runner-up'} and the effect over the baseline.")
        rationale, _ = narrate(self.ctx.llm, "Explain why the supervisor chose this pipeline.", facts, fallback)
        rec = Recommendation(agent=self.name, decision_point="model_choice", recommendation=chosen_id,
                             rationale=rationale, evidence=evidence, confidence=round(conf["confidence"], 4),
                             confidence_method="bootstrap", verified=bool(winning) and all(c.verified for c in winning),
                             claims=winning)
        if positions:
            pos_rec = Recommendation(agent=self.name, decision_point="debate_positions",
                                     recommendation="; ".join(f"{p.agent}: {p.preferred}" for p in positions),
                                     rationale=f"{verifier.get('verified', 0)} of {verifier.get('total', 0)} claims verified",
                                     evidence=[e for p in positions for c in p.reasons[:1] for e in c.evidence][:8] or [ev],
                                     confidence=1.0, confidence_method="rule")
            self.ctx.ledger.add(pos_rec, options=[p.agent for p in positions], chosen="positions recorded",
                                decided_by="agent_rule")
        self.ctx.ledger.add(rec, options=[c.id for c in survivors], chosen=chosen_id, decided_by="supervisor",
                            note=("forced tiebreak" if forced else None))
        transcript = transcript + [DebateTurn(round=transcript[-1].round if transcript else 0, agent="supervisor",
                                              kind="decision", text=f"final: {chosen_id} (confidence {conf['confidence']:.2f})",
                                              data={"confidence": conf})]
        self.finished(f"{chosen_id} chosen ({rule})")
        return DebateOutcome(chosen=chosen_id, runner_up=runner.id if runner else None, rule=rule,
                             forced_tiebreak=forced, positions=positions, transcript=transcript,
                             verifier={**verifier, "confidence_components": conf}, notes=notes, recommendation=rec)
