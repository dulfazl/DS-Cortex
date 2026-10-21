import json

import numpy as np
import pandas as pd
import pytest

from agents.base import AgentContext
from agents.pipeline_critic.core import PipelineCriticAgent
from app.llm import FakeLLM
from app.schemas import CandidatePipeline, Check, Claim, CVScores, Evidence, TaskPlan
from tools.tool_log import ToolLog
from tools.verify_tools import numbers_supported, verify_claims
from workflows.debate_graph import run_debate
from workflows.resolution_table import DebateState, resolve

PLAN = TaskPlan(task_family="classification", target="y", metric="f1_macro", secondary_metrics=["roc_auc"],
                rationale="t", confidence=0.9)


@pytest.fixture
def noisy_df():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 6))
    y = ((X[:, 0] + rng.normal(0, 1.5, n)) > 0).astype(int)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(6)]).assign(y=y)


def _cand(ctx, cid, model, mean, std, params=None, interpretable=False):
    spec = {"model": model, "params": params or {}, "prep": {"scale": model == "logreg"}, "balance": None}
    folds = [mean + d for d in (-std, std, 0.0, -std / 2, std / 2)]
    res, ev = ctx.tool("sklearn_tools.evaluate_pipeline", lambda: {"primary": "f1_macro", "primary_mean": mean,
                                                                   "primary_std": std, "per_fold": folds}, {"candidate": cid})
    cv = CVScores(primary="f1_macro", primary_mean=mean, primary_std=std, per_fold=folds, fit_time=0.1 if interpretable else 1.0,
                  n_splits=5)
    return CandidatePipeline(id=cid, name=cid, model=model, spec=spec, cv=cv, interpretable=interpretable, eval_run_id=ev.run_id)


def test_validity_beats_raw_performance(noisy_df):
    ctx = AgentContext.offline("debate1", persist=True)
    a = _cand(ctx, "A_rf", "random_forest", 0.90, 0.02, {"n_estimators": 50, "max_depth": None, "min_samples_leaf": 1})
    b = _cand(ctx, "B_logreg", "logreg", 0.85, 0.02, interpretable=True)
    a.objections = PipelineCriticAgent(ctx).review(a, noisy_df, PLAN, 0.5, only={"overfit_check"})
    assert a.objections and a.objections[0].check == "overfit_check"
    res = run_debate(ctx, [a, b], PLAN, None, noisy_df, 0.5)
    assert res.chosen == "B_logreg"
    assert any(d.rule == "validity_vs_performance" for d in res.decisions)
    assert any(t.kind == "rebuttal" for t in res.transcript)          # the Architect got to answer
    assert res.forced_tiebreak                                          # objection stood through both rounds
    assert sum(1 for t in res.transcript if t.kind == "rebuttal") <= 2  # never loops


def test_fabricated_claims_are_rejected():
    log = ToolLog("v", persist=False)
    res, ev = log.run("stats_tools.vif_report", lambda: {"mean_vif": 3.2, "columns": {"a": {"vif": 12.0}}})
    good = Claim(text="a has VIF 12", evidence=[ev], checks=[Check(run_id=ev.run_id, path="columns.a.vif", op="approx", value=12.0)])
    wrong_value = Claim(text="a has VIF 40", evidence=[ev], checks=[Check(run_id=ev.run_id, path="columns.a.vif", op="approx", value=40)])
    fake_run = Claim(text="b has VIF 50", evidence=[Evidence(tool="stats_tools.vif_report", result={"b": 50}, run_id="nope")],
                     checks=[Check(run_id="nope", path="b", op="approx", value=50)])
    no_check = Claim(text="trust me", evidence=[ev])
    tampered = Claim(text="copy edited", evidence=[Evidence(tool=ev.tool, result={"mean_vif": 99}, run_id=ev.run_id)],
                     checks=[Check(run_id=ev.run_id, path="mean_vif", op="approx", value=99)])
    counts = verify_claims([good, wrong_value, fake_run, no_check, tampered], log)
    assert good.verified and not wrong_value.verified and not fake_run.verified and not no_check.verified
    assert not tampered.verified          # checks run against the logged result, not the claim's copy
    assert counts == {**counts, "total": 5, "verified": 1, "rejected": 4}


def test_llm_fabricated_debate_claim_counted(noisy_df):
    def reply(prompt, system):
        return json.dumps({"preferred": "A_rf", "claims": [
            {"text": "A_rf has the lowest error", "run_id": "deadbeef", "path": "primary_mean", "op": "approx", "value": 0.99}]})
    ctx = AgentContext.offline("debate2", persist=True)
    ctx.llm = FakeLLM(reply=reply)
    a = _cand(ctx, "A_rf", "random_forest", 0.80, 0.02, {"n_estimators": 50})
    b = _cand(ctx, "B_logreg", "logreg", 0.85, 0.02, interpretable=True)
    res = run_debate(ctx, [a, b], PLAN, None, noisy_df, 0.5)
    assert res.verifier["rejected_llm"] >= 4 and res.verifier["total_llm"] >= 4
    assert res.chosen == "B_logreg"        # the fabricated preference was not adopted


def test_explainability_rule_and_tiebreak():
    ctx = AgentContext.offline("debate3", persist=True)
    a = _cand(ctx, "A_xgb", "xgboost", 0.90, 0.01)
    b = _cand(ctx, "B_tree", "decision_tree", 0.86, 0.01, interpretable=True)
    from agents.business.core import BusinessInsightAgent
    from agents.ml_architect.core import MLArchitectAgent
    plan = PLAN.model_copy(update={"explainability_required": True})
    pos = {"ml_architect": MLArchitectAgent(ctx).position([a, b], plan), "business": BusinessInsightAgent(ctx).position([a, b], plan)}
    for p in pos.values():
        verify_claims(p.reasons, ctx.log)
    chosen, decisions = resolve(DebateState(positions=pos, candidates={"A_xgb": a, "B_tree": b}, plan=plan))
    assert chosen == "B_tree" and decisions[-1].rule == "explainability_vs_metric"
    chosen, decisions = resolve(DebateState(positions=pos, candidates={"A_xgb": a, "B_tree": b}, plan=plan,
                                            explainability_waived=True))
    assert decisions and decisions[-1].rule == "architect_vs_business"


def test_narration_number_check():
    ok, bad = numbers_supported("The model scores 0.873 F1, beating the baseline of 0.41.", {"f1": 0.8731, "base": 0.4102})
    assert ok and not bad
    ok, bad = numbers_supported("Accuracy improved by 37% to 0.95.", {"f1": 0.8731})
    assert not ok and set(bad) == {"37%", "0.95"}
