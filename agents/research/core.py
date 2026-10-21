"""Research agent, scoped small: retrieves a few notes on established practice for this kind of
data and flags divergence. Context, not a literature review."""
from __future__ import annotations

from agents.base import BaseAgent, cv_claim
from agents.ml_architect.core import MLArchitectAgent
from app.schemas import Check, Claim, DebatePosition, StatisticalProfile, TaskPlan
from memory import vector_store as vs

PRACTICE = {
    "gbm_tabular": {"gradient_boosting", "xgboost", "lightgbm"},
    "regularized_linear": {"logreg", "logreg_l1", "ridge", "lasso"},
    "rf_default": {"random_forest"},
    "ts_baseline": {"seasonal_naive", "arima", "gbm_lags"},
    "kmeans_default": {"kmeans"},
    "clustering_methods": {"gmm", "dbscan", "agglomerative"},
    "iforest_default": {"isolation_forest"},
    "offline_rl": {"greedy_ridge", "greedy_rf", "greedy_gbm"},
}


def practice_for(plan: TaskPlan, signals: dict) -> tuple[str, str]:
    fam, n = plan.task_family, signals.get("n_rows", 1000)
    if fam in ("classification", "regression"):
        if n < 300 or signals.get("high_vif"):
            return "regularized_linear", f"{fam} with few rows or collinear features: regularized linear models"
        return "gbm_tabular", f"tabular {fam} with {n} rows: gradient-boosted trees"
    if fam == "time_series":
        return "ts_baseline", "forecasting: simple seasonal benchmarks are hard to beat"
    if fam == "clustering":
        return "kmeans_default", "clustering of roughly spherical groups: k-means as the reference method"
    if fam == "anomaly_detection":
        return "iforest_default", "unsupervised anomaly detection: isolation forest as the reference method"
    return "offline_rl", "offline policy learning: reward models valued with inverse propensity weighting"


class ResearchAgent(BaseAgent):
    name = "research"

    def context(self, plan: TaskPlan, profile: StatisticalProfile | None, extra: str = "") -> dict:
        s = profile.signals if profile else {}
        words = [plan.task_family.replace("_", " "), "tabular"]
        if plan.imbalanced:
            words += ["imbalanced", "precision recall"]
        if s.get("high_vif"):
            words += ["multicollinearity", "regularization"]
        if s.get("small_sample"):
            words += ["small sample", "overfitting"]
        if s.get("heavy_skew"):
            words += ["skew", "power transform"]
        if plan.explainability_required:
            words += ["interpretable", "explainability"]
        res, ev = self.ctx.tool("memory.vector_store.retrieve", vs.retrieve, {"query": " ".join(words + [extra])},
                                " ".join(words + [extra]), 5)
        return {"result": res, "evidence": ev}

    def position(self, survivors, plan: TaskPlan, profile: StatisticalProfile | None, round_no: int = 0) -> tuple[DebatePosition, list[str]]:
        signals = profile.signals if profile else {}
        practice, why = practice_for(plan, signals)
        ctx = self.context(plan, profile, extra=practice.replace("_", " "))
        hits = ctx["result"]["hits"]
        note = next((h for h in hits if h["practice"] == practice), None) or next(iter(vs.by_practice(practice)), None)
        ranked = MLArchitectAgent.rank(survivors, plan.metric)
        in_practice = [c for c in ranked if c.model in PRACTICE.get(practice, set())]
        pref = in_practice[0] if in_practice else ranked[0]
        reasons = []
        if note and note["id"] in ctx["result"]["ids"]:
            reasons.append(Claim(kind="practice", text=f"established practice for {why} (see {note['cite']}: {note['title']})",
                                 evidence=[ctx["evidence"]],
                                 checks=[Check(run_id=ctx["evidence"].run_id, path="ids", op="contains", value=note["id"])]))
        c = cv_claim(self.ctx.log, pref)
        if c:
            reasons.append(c)
        notes = []
        if ranked[0].model not in PRACTICE.get(practice, set()):
            notes.append(f"Research note: the top-scoring candidate ({ranked[0].model}) differs from established practice "
                         f"for this data ({why}). This is informational only.")
        if plan.task_family == "time_series":
            sn = next((c for c in survivors if c.model == "seasonal_naive"), None)
            if sn and ranked[0].model != "seasonal_naive":
                notes.append("Research note: forecasts should be reported against the seasonal naive benchmark, as here.")
        return DebatePosition(agent=self.name, preferred=pref.id, reasons=reasons, confidence=0.6,
                              confidence_method="rule", round=round_no), notes
