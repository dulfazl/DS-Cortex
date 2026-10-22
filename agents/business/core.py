"""Business Insight: turns findings into stakeholder language and argues for what the business
needs: an interpretable model when the client asked for one, otherwise the cheapest model that is
statistically as good as the best."""
from __future__ import annotations

from agents.base import BaseAgent, cv_claim
from app.llm import narrate
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
            if interp:
                pref = interp[0]
                reasons = [Claim(kind="interpretability",
                                 text=f"the client asked for an explainable model; {pref} ({cands[pref]['model']}) is interpretable",
                                 evidence=[ev], checks=[Check(run_id=ev.run_id, path=f"candidates.{pref}.interpretable", op="true")])]
            else:
                pref = table["best"]
                reasons = [Claim(kind="interpretability_unavailable",
                                 text=f"the client asked for an explainable model, but no surviving candidate is interpretable; "
                                      f"{pref} is the best-scoring option and needs SHAP explanations",
                                 evidence=[ev], checks=[Check(run_id=ev.run_id, path=f"candidates.{pref}.interpretable", op="false")])]
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

    def stakeholder_summary(self, plan: TaskPlan, test: dict, importance: dict | None, profile, audit,
                            ledger_steps: dict[str, int]) -> dict:
        """KPIs and recommendations in plain language. Each recommendation cites an upstream finding
        (a ledger step); claims are softened where the Statistician flagged assumption violations."""
        self.started("writing the stakeholder summary")
        metric, score, base = plan.metric, test["score"], test["baseline"]["score"]
        k, _ = self.ctx.tool("business_tools.kpis", bt.kpis, {"metric": metric}, metric, score, base)
        gain, rel = k["gain"], k["relative_gain"]
        kpis = [{"name": f"held-out {metric}", "value": score, "baseline": base,
                 "lift": f"{rel:+.1%} relative to the baseline" if rel is not None else None}]
        vs = test.get("vs_baseline") or {}
        # no signal: the model is not better than the baseline by more than chance, so its importances describe noise
        no_signal = gain <= 0 or vs.get("meaningful") is False
        drivers = [] if no_signal else list((importance or {}).get("importance", {}).items())[:3]
        target = plan.target or "the outcome"
        recs = []
        if no_signal:
            recs.append({"text": f"Do not use this model for decisions about {target}. To predict it, add information that is "
                                 "known before the outcome and plausibly related to it: the columns in this file carry little or none.",
                         "cites": f"step {ledger_steps.get('final_test_score', '?')} (held-out comparison with the baseline)"})
        if drivers:
            recs.append({"text": "The outcome is driven mostly by " + ", ".join(d for d, _ in drivers)
                         + ". Focus data quality and business attention there.",
                         "cites": f"step {ledger_steps.get('final_test_score', '?')} (final model) and the importance analysis"})
        for f in (audit.findings if audit else [])[:3]:
            if f.severity in ("blocker", "high"):
                if f.issue_type in ("possible_proxy", "target_leakage"):
                    text = (f"Check when '{f.column}' is recorded: it predicts the outcome very strongly on its own. "
                            "If it is only known after the outcome, it must not be used.")
                else:
                    text = (f"Data issue to fix at the source: {f.issue_type.replace('_', ' ')} in "
                            f"{f.column or 'the rows'} ({f.recommended_action}).")
                recs.append({"text": text,
                             "cites": f"audit finding on {f.column or 'rows'} (step {ledger_steps.get('audit:' + f.issue_type + (':' + f.column if f.column else ''), '?')})"})
        caveats = []
        if no_signal and gain > 0 and vs.get("applicable"):
            lo, hi = vs["ci95"]
            caveats.append(f"The gain over the baseline ({vs['gain']:+.3f}) is within what chance produces on {vs['n_rows']} "
                           f"held-out rows (95% interval {lo:+.3f} to {hi:+.3f}), so no main factors are named.")
        flags = {fl.issue for fl in (profile.flags if profile else [])}
        if "heteroscedasticity" in flags:
            caveats.append("Prediction errors are larger for some kinds of rows, so treat single predictions as approximate.")
        if "small_sample" in flags:
            caveats.append("The dataset is small: the measured performance could shift noticeably on new data.")
        if "weak_signal" in flags:
            caveats.append("Each input on its own says little about the outcome, so gains over the baseline are modest.")
        if "multicollinearity" in flags:
            caveats.append("Several inputs move together, so do not read the effect of any single input in isolation.")
        if gain <= 0:
            caveats.append("The chosen model did not beat the simple baseline on the held-out data, so no main factors are named.")
        facts = {"metric": metric, "score": score, "baseline": base, "relative_lift": rel,
                 "beats_baseline_beyond_chance": not no_signal, "gain_ci95": vs.get("ci95"),
                 "drivers": [d for d, _ in drivers], "caveats": caveats, "recommendations": [r["text"] for r in recs]}
        if no_signal:
            others = [c for c in caveats if "chance produces" not in c and "did not beat" not in c]   # shown as caveats already
            fallback = (f"On data it never saw, the model scores {score:.3f} {metric} against {base:.3f} for a simple baseline: "
                        f"no better than chance, so this file does not show what drives {target}."
                        + (" " + " ".join(others) if others else ""))
            task = ("Write a short summary for a business stakeholder. The model is no better than a simple baseline beyond "
                    "chance: say so plainly in the first sentence, name no drivers or factors, and do not suggest acting on it.")
        else:
            fallback = (f"On data it never saw, the model scores {score:.3f} {metric} against {base:.3f} for a simple baseline"
                        + (f" ({rel:+.1%})." if rel is not None else ".") + (" " + " ".join(caveats) if caveats else ""))
            task = "Write a short summary for a business stakeholder. Soften any claim the caveats qualify."
        text, source = narrate(self.ctx.llm, task, facts, fallback, depth="beginner")
        self.finished("stakeholder summary ready")
        return {"headline": text, "source": source, "kpis": kpis, "recommendations": recs, "caveats": caveats,
                "signal": not no_signal}
