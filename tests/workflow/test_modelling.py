import numpy as np
import pytest

from agents.base import AgentContext
from agents.ml_architect.core import MLArchitectAgent
from agents.pipeline_critic.core import PipelineCriticAgent
from app.schemas import TaskPlan
from tools import pipeline_core as pc
from tools.pandas_tools import coerce_types, load_csv
from workflows.modelling import critic_loop

PLANS = {
    "classification.csv": TaskPlan(task_family="classification", target="churn", metric="f1_macro",
                                   secondary_metrics=["roc_auc"], rationale="t", confidence=0.9),
    "regression.csv": TaskPlan(task_family="regression", target="price", metric="rmse", secondary_metrics=["r2", "mae"],
                               rationale="t", confidence=0.9),
    "clustering.csv": TaskPlan(task_family="clustering", metric="silhouette", secondary_metrics=["davies_bouldin"],
                               rationale="t", confidence=0.9),
}


@pytest.mark.parametrize("name", list(PLANS))
def test_architect_returns_four_scored_candidates(fixtures_dir, name):
    df = coerce_types(load_csv(fixtures_dir / name))
    ctx = AgentContext.offline(persist=True)
    cands = MLArchitectAgent(ctx, folds=5, repeats=1).propose(df, PLANS[name], None)
    assert len(cands) >= 4
    assert all(c.cv and c.cv.n_splits >= 3 and np.isfinite(c.cv.primary_mean) for c in cands)
    assert all(ctx.log.get(c.eval_run_id) for c in cands)
    assert [e.decision_point for e in ctx.ledger.entries()] == ["candidate_generation"]


def test_critic_catches_injected_leak(fixtures_dir):
    df = coerce_types(load_csv(fixtures_dir / "classification.csv"))
    df["leak"] = df["churn"] + np.random.default_rng(0).normal(0, 0.05, len(df))
    plan = PLANS["classification.csv"]
    ctx = AgentContext.offline(persist=True)
    arch = MLArchitectAgent(ctx, folds=5, repeats=1)
    cands = arch.propose(df, plan, None)
    best = arch.rank(cands, plan.metric)[0]
    objs = PipelineCriticAgent(ctx).review(best, df, plan, arch.baseline["primary_mean"], top_feature="leak",
                                           only={"leakage_probe"})
    assert any(o.check == "leakage_probe" and o.severity in ("high", "blocker") for o in objs)


def test_revision_loop_respects_round_cap(fixtures_dir):
    df = coerce_types(load_csv(fixtures_dir / "classification.csv"))
    plan = PLANS["classification.csv"]
    ctx = AgentContext.offline(persist=True)
    arch = MLArchitectAgent(ctx, folds=5, repeats=1)
    cands = arch.propose(df, plan, None)
    out = critic_loop(ctx, arch, PipelineCriticAgent(ctx), cands, df, plan, arch.baseline["primary_mean"], top_k=3)
    rounds = [e for e in ctx.ledger.entries() if e.decision_point.startswith("critic_round_")]
    assert len(rounds) <= 3
    assert any(c.status == "active" for c in out)
    revised = [c for c in out if c.origin == "revision"]
    assert all(c.parent for c in revised)
