import pytest

from agents.base import AgentContext
from agents.router.core import RouterAgent
from app.schemas import ClarifyingQuestion, TaskPlan
from tools.pandas_tools import coerce_types, load_csv, quick_profile

CASES = [
    ("classification.csv", "Predict whether a customer will churn", "classification", "churn"),
    ("regression.csv", "Estimate the sale price of each house", "regression", "price"),
    ("clustering.csv", "Segment our customers into groups", "clustering", None),
    ("anomaly.csv", "Find unusual or fraudulent transactions", "anomaly_detection", None),
    ("timeseries.csv", "Forecast sales for next month", "time_series", "sales"),
    ("bandit.csv", "Learn which offer to show each user to maximize reward", "offline_rl", "reward"),
]
VAGUE = [
    ("classification.csv", "Analyze this data and tell me something useful"),
    ("two_targets.csv", "Look at price and rent for these homes"),
]


def _route(fixtures_dir, name, statement, answer=None):
    path = fixtures_dir / name
    profile = quick_profile(path)
    df = coerce_types(load_csv(path))
    return RouterAgent(AgentContext.offline()).plan(statement, profile, df, answer=answer), profile


@pytest.mark.parametrize("name,statement,family,target", CASES)
def test_router_picks_family(fixtures_dir, name, statement, family, target):
    plan, _ = _route(fixtures_dir, name, statement)
    assert isinstance(plan, TaskPlan), plan
    assert plan.task_family == family
    assert plan.target == target
    assert plan.confidence >= 0.7 and plan.metric


@pytest.mark.parametrize("name,statement", VAGUE)
def test_vague_statements_ask_instead_of_guessing(fixtures_dir, name, statement):
    q, _ = _route(fixtures_dir, name, statement)
    assert isinstance(q, ClarifyingQuestion)
    assert len(q.options) >= 2


def test_metric_fixed_per_family(fixtures_dir):
    plan, _ = _route(fixtures_dir, "regression.csv", "Estimate the sale price of each house")
    assert plan.metric == "rmse" and plan.secondary_metrics == ["r2", "mae"]
    plan, _ = _route(fixtures_dir, "timeseries.csv", "Forecast sales for next month")
    assert plan.metric == "smape" and plan.time_column == "month"


def test_rl_honesty_check(fixtures_dir):
    q, profile = _route(fixtures_dir, "classification.csv", "Find the best action policy to reduce churn")
    assert isinstance(q, ClarifyingQuestion) and q.reason == "rl_mismatch"
    assert "does not fit" in q.question and "classification" in q.question
    answer = RouterAgent.parse_answer(q, q.options[0], [c["name"] for c in profile["column_info"]])
    plan, _ = _route(fixtures_dir, "classification.csv", "Find the best action policy to reduce churn", answer)
    assert isinstance(plan, TaskPlan) and plan.task_family == "classification" and plan.target == "churn"


def test_clarification_answer_resolves(fixtures_dir):
    q, profile = _route(fixtures_dir, "two_targets.csv", "Look at price and rent for these homes")
    answer = RouterAgent.parse_answer(q, "rent", [c["name"] for c in profile["column_info"]])
    plan, _ = _route(fixtures_dir, "two_targets.csv", "Look at price and rent for these homes", answer)
    assert plan.task_family == "regression" and plan.target == "rent"
    q, profile = _route(fixtures_dir, "classification.csv", "Analyze this data and tell me something useful")
    answer = RouterAgent.parse_answer(q, "Find segments (clustering)", [c["name"] for c in profile["column_info"]])
    plan, _ = _route(fixtures_dir, "classification.csv", "Analyze this data", answer)
    assert plan.task_family == "clustering" and plan.target is None


def test_router_writes_ledger(fixtures_dir):
    ctx = AgentContext.offline()
    path = fixtures_dir / "classification.csv"
    RouterAgent(ctx).plan("Predict whether a customer will churn", quick_profile(path), load_csv(path))
    points = [e.decision_point for e in ctx.ledger.entries()]
    assert points == ["task_family", "primary_metric"]
