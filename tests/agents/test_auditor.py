import numpy as np
import pandas as pd
import pytest

from agents.base import AgentContext
from agents.data_auditor.core import DataAuditorAgent
from agents.statistician.core import StatisticianAgent
from app.schemas import TaskPlan
from tools import pipeline_core as pc
from tools.pandas_tools import coerce_types


def messy_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n).astype(float)
    income = rng.lognormal(10.5, 0.6, n)
    plan = rng.choice(["Basic", "basic", "PRO ", "pro", "team"], n)
    logit = -1 + 0.04 * (50 - age) + 0.8 * np.isin(plan, ["Basic", "basic"])
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "customer_id": np.arange(10000, 10000 + n),
        "age": age, "income": income, "plan": plan,
        "visits": rng.poisson(5, n).astype(str),          # numbers stored as text
        "country": "IN",                                    # constant
        "churn": y,
    })
    df.loc[rng.choice(n, 8, replace=False), "age"] = -5    # impossible values
    df.loc[df["age"] > 60, "income"] = np.nan              # MAR-like missingness (depends on age)
    df.loc[rng.choice(n, 5, replace=False), "income"] = 5e6  # outliers
    df = pd.concat([df, df.iloc[:12]], ignore_index=True)  # exact duplicates
    return df


PLAN = TaskPlan(task_family="classification", target="churn", metric="f1_macro", secondary_metrics=["roc_auc"],
                rationale="test", confidence=0.9)


def _audit(df):
    ctx = AgentContext.offline()
    return DataAuditorAgent(ctx).audit(coerce_types(df), PLAN), ctx


def test_auditor_finds_the_visible_issues():
    report, ctx = _audit(messy_frame())
    issues = {(f.issue_type, f.column) for f in report.findings}
    assert ("id_like", "customer_id") in issues
    assert ("constant", "country") in issues
    assert ("impossible_values", "age") in issues
    assert ("mixed_case", "plan") in issues
    assert ("duplicates", None) in issues
    assert any(i == "missing_values" and c == "income" for i, c in issues)
    miss = next(f for f in report.findings if f.issue_type == "missing_values" and f.column == "income")
    assert "MCAR" not in miss.rationale            # missingness depends on age
    assert not report.blockers
    actions = {(a.action, a.column) for a in report.cleaning_plan}
    assert ("drop_column", "customer_id") in actions and ("drop_duplicates", None) in actions
    # every finding carries real evidence that exists in the tool log
    for f in report.findings:
        assert f.evidence and ctx.log.get(f.evidence[0].run_id) is not None


def test_injected_leak_is_a_blocker():
    df = messy_frame()
    rng = np.random.default_rng(1)
    df["outcome_score"] = df["churn"] + rng.normal(0, 0.05, len(df))
    report, _ = _audit(df)
    assert "outcome_score" in report.blockers
    f = next(f for f in report.findings if f.column == "outcome_score")
    assert f.severity == "blocker" and f.confidence >= 0.97 and f.confidence_method == "effect_size"


def test_evidence_matches_manual_calculation():
    df = messy_frame()
    report, ctx = _audit(df)
    dup = next(f for f in report.findings if f.issue_type == "duplicates")
    logged = ctx.log.get(dup.evidence[0].run_id)["result"]
    assert logged["exact_count"] == int(df.duplicated().sum()) == 12
    miss = next(f for f in report.findings if f.issue_type == "missing_values" and f.column == "income")
    logged = ctx.log.get(miss.evidence[0].run_id)["result"]["columns"]["income"]
    assert logged["missing_count"] == int(df["income"].isna().sum())
    neg = next(f for f in report.findings if f.issue_type == "impossible_values")
    assert ctx.log.get(neg.evidence[0].run_id)["result"]["columns"]["age"]["negative_values"] == int((df["age"] < 0).sum())


def test_cleaning_plan_applies_and_logs():
    df = coerce_types(messy_frame())
    report, _ = _audit(df)
    clean, log = pc.apply_cleaning_plan(df, [a.model_dump() for a in report.cleaning_plan])
    assert "customer_id" not in clean and "country" not in clean
    assert len(clean) == len(df) - 12          # the 12 copied rows are gone, nothing else
    assert log[0]["action"] == "drop_duplicates"  # duplicates are judged on the raw rows
    assert (clean["age"].dropna() >= 0).all()
    assert set(clean["plan"].unique()) <= {"basic", "pro", "team"}
    assert any(e["action"] == "drop_duplicates" for e in log)


def test_statistician_profile_and_fingerprint():
    df = coerce_types(messy_frame())
    report, ctx = _audit(df)
    clean, _ = pc.apply_cleaning_plan(df, [a.model_dump() for a in report.cleaning_plan])
    prof = StatisticianAgent(ctx).profile(clean, PLAN, report)
    assert len(prof.fingerprint) == 12 and all(0 <= v <= 1 for v in prof.fingerprint)
    assert any(f.issue == "heavy_skew" for f in prof.flags)          # lognormal income
    assert prof.signals["n_rows"] == len(clean)
    assert prof.guidance
