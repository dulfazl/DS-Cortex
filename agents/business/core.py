"""Business Insight: turns findings into stakeholder language and argues for what the business
needs: an interpretable model when the client asked for one, otherwise the cheapest model that is
statistically as good as the best."""
from __future__ import annotations

from agents.base import BaseAgent, cv_claim
from app.schemas import Check, Claim, DebatePosition, TaskPlan
from tools import business_tools as bt
from tools.common import clip01
from tools.sklearn_tools import bounded_metric


class BusinessInsightAgent(BaseAgent):
    name = "business"

    def position(self, survivors, plan: TaskPlan, round_no: int = 0) -> DebatePosition:
        rows = [{"id": c.id, "model": c.model, "interpretable": c.interpretable, "fit_time": c.cv.fit_time,
                 "mean": c.cv.primary_mean, "std": c.cv.primary_std} for c in survivors]
        table, ev = self.ctx.tool("business_tools.candidate_table", bt.candidate_table, {"metric": plan.metric},
                                  rows, plan.metric)
        cands = table["candidates"]
        tol = 0.1 if bounded_metric(plan.metric) else 0.1 * abs(cands[table["best"]]["primary_mean"])
        if plan.explainability_required:
            interp = sorted([cid for cid, r in cands.items() if r["interpretable"]], key=lambda cid: cands[cid]["gap_to_best"])
            pref = interp[0] if interp else table["best"]
            reasons = [Claim(kind="interpretability", text=f"the client asked for an explainable model; {pref} ({cands[pref]['model']}) is interpretable",
                             evidence=[ev], checks=[Check(run_id=ev.run_id, path=f"candidates.{pref}.interpretable", op="true")])]
        else:
            close = [cid for cid, r in cands.items() if r["within_1sd"]]
            pref = min(close, key=lambda cid: cands[cid]["fit_time"])
            reasons = [Claim(kind="cost", text=f"{pref} is within one standard deviation of the best and the cheapest to train "
                                  f"({cands[pref]['fit_time']:.2f} s across folds)", evidence=[ev],
                             checks=[Check(run_id=ev.run_id, path=f"candidates.{pref}.within_1sd", op="true"),
                                     Check(run_id=ev.run_id, path=f"candidates.{pref}.fit_time", op="approx",
                                           value=cands[pref]["fit_time"], tol=1e-3)])]
        c = cv_claim(self.ctx.log, next(s for s in survivors if s.id == pref))
        if c:
            reasons.append(c)
        conf = clip01(1 - cands[pref]["gap_to_best"] / tol) if tol else 0.5
        return DebatePosition(agent=self.name, preferred=pref, reasons=reasons, confidence=round(conf, 4),
                              confidence_method="effect_size", round=round_no)
