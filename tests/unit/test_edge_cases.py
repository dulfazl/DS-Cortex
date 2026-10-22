"""Files that break real apps. Each must end in a readable message or a clarifying question,
never a stack trace; the audit tools must never crash and must return JSON-ready results."""
import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agents.base import AgentContext
from agents.data_auditor.core import DataAuditorAgent
from app.schemas import TaskPlan
from app.session import RunFlags, store
from tools import pandas_tools as pt
from tools import pipeline_core as pc
from tools.common import DataError
from workflows.pipeline_graph import PipelineRun


def _write(tmp_path, name, content, encoding="utf-8"):
    p = tmp_path / name
    p.write_bytes(content.encode(encoding) if isinstance(content, str) else content)
    return p


@pytest.mark.parametrize("content,message", [
    ("", "empty"),
    ("a,b,c\n", "no data rows"),
    ("   \n\n", "empty"),
])
def test_unusable_files_give_readable_errors(tmp_path, content, message):
    with pytest.raises(DataError, match=message):
        pt.load_csv(_write(tmp_path, "x.csv", content))


def test_odd_but_valid_files_load(tmp_path):
    semi = pt.load_csv(_write(tmp_path, "semi.csv", "a;b;c\n1;2;3\n4;5;6\n"))
    assert list(semi.columns) == ["a", "b", "c"] and len(semi) == 2
    latin = pt.load_csv(_write(tmp_path, "latin.csv", "név,érték\nÁrpád,1\nÉva,2\n", encoding="latin-1"))
    assert list(latin.columns) == ["név", "érték"]
    bom = pt.load_csv(_write(tmp_path, "bom.csv", b"\xef\xbb\xbfid,v\n1,2\n2,3\n"))
    assert list(bom.columns) == ["id", "v"]
    dup = pt.load_csv(_write(tmp_path, "dup.csv", "x,x,,y\n1,2,3,4\n5,6,7,8\n"))
    assert len(set(dup.columns)) == 4 and dup.columns[2].startswith("column_")
    one = pt.load_csv(_write(tmp_path, "one.csv", "only\n1\n"))
    assert one.shape == (1, 1)
    tabs = pt.load_csv(_write(tmp_path, "tabs.csv", "a\tb\n1\t2\n"))
    assert list(tabs.columns) == ["a", "b"]


def test_types_stored_as_text(tmp_path):
    rows = ["when,amount,label"] + [f"{d},\"{1000 + i:,}\",{i % 2}" for i, d in
                                    enumerate(["2024-01-05", "05/02/2024", "March 3 2024", "2024-04-01"] * 10)]
    df = pt.load_csv(_write(tmp_path, "t.csv", "\n".join(rows) + "\n"))
    kinds = pt.type_summary(df)
    assert kinds["amount"]["stored_as_text"] == "numeric"
    assert kinds["when"]["kind"] == "datetime"
    co = pt.coerce_types(df)
    assert pd.api.types.is_numeric_dtype(co["amount"]) and co["amount"].iloc[1] == 1001


def test_wide_file_profiles(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(60, 1000)), columns=[f"c{i}" for i in range(1000)])
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)
    prof = pt.quick_profile(p)
    assert prof["columns"] == 1000 and len(prof["column_info"]) == 1000


def _run(tmp_path, df, statement, name="d.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    s = store.create(str(p), statement, flags=RunFlags(memory=False, tuning=False, explain=False))
    return PipelineRun(s.id, interactive=False).run()


def test_single_class_target_fails_readably(tmp_path):
    df = pd.DataFrame({"x": range(50), "z": np.random.default_rng(0).normal(size=50), "y": ["yes"] * 50})
    s = _run(tmp_path, df, "Predict y")
    assert s.status == "failed"
    assert "Traceback" not in (s.error or "") and s.error


def test_all_missing_and_constant_columns_are_dropped(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=120), "empty": [None] * 120, "const": 7,
                       "y": (rng.normal(size=120) > 0).astype(int)})
    s = _run(tmp_path, df, "Predict y from x")
    assert s.status == "completed", s.error
    assert "empty" not in s.stage_outputs["clean"]["columns"] and "const" not in s.stage_outputs["clean"]["columns"]


def test_vague_statement_without_answer_fails_readably(tmp_path):
    df = pd.DataFrame({"a": range(40), "b": range(40, 80)})
    s = _run(tmp_path, df, "do something")
    assert s.status == "failed" and "clarification" in s.error


frames = st.integers(min_value=5, max_value=40).flatmap(lambda n: st.tuples(
    st.lists(st.one_of(st.none(), st.floats(-1e6, 1e6, allow_nan=False)), min_size=n, max_size=n),
    st.lists(st.one_of(st.none(), st.sampled_from(["a", "B", "b ", "c", "ünï"])), min_size=n, max_size=n),
    st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n),
))


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(frames)
def test_audit_tools_never_crash(data):
    num, cat, y = data
    df = pd.DataFrame({"num": num, "cat": cat, "y": y})
    for fn, args in [(pt.missing_report, (df, "y")), (pt.type_inference, (df,)), (pt.outlier_report, (df, ["y"])),
                     (pt.duplicate_report, (df,)), (pt.inconsistency_report, (df,))]:
        json.dumps(fn(*args))
    if df["y"].nunique() == 2:
        json.dumps(pt.leakage_report(df, "y", "classification"))
    plan = TaskPlan(task_family="classification", target="y", metric="f1_macro", rationale="t", confidence=0.9)
    report = DataAuditorAgent(AgentContext.offline()).audit(df, plan)
    clean, log = pc.apply_cleaning_plan(df, [a.model_dump() for a in report.cleaning_plan])
    assert isinstance(clean, pd.DataFrame)


def test_row_cap_samples_and_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_ROWS", "120")
    from app.config import reset_settings
    reset_settings()
    from app.ledger import Ledger
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=400), "y": (rng.normal(size=400) > 0).astype(int)})
    s = _run(tmp_path, df, "Predict y from x")
    assert s.status == "completed", s.error
    assert s.stage_outputs["clean"]["rows"] == 120
    assert any(e.decision_point == "row_cap" for e in Ledger(s.id).entries())
    assert any(a.action == "sample_rows" for a in s.audit_report.cleaning_plan)   # so the exported script reproduces it


def test_multiclass_imbalanced_target_with_dates_completes(tmp_path):
    """The shape of a real upload that stalled: four classes (one at 50%), a date column from a single year,
    and name-like categoricals. It is routed as imbalanced classification, whose secondary metric used to be NaN
    for more than two classes, and the unscaled year made the L1 logistic regression crawl."""
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({"placed_at": pd.date_range("2026-01-01", periods=n, freq="37min").strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "first": rng.choice(list("ABCDEFGHIJ"), n), "country": rng.choice(["AR", "BR", "CA", "FR"], n),
                       "items": rng.integers(1, 6, n), "total": rng.integers(500, 60000, n),
                       "status": rng.choice(["paid", "refunded", "cancelled", "shipped"], n, p=[0.5, 0.17, 0.17, 0.16])})
    s = _run(tmp_path, df, "Predict status")
    assert s.status == "completed", s.error
    assert s.task_plan.metric == "pr_auc"
    assert store.get(s.id).status == "completed"   # the stored session loads
    for c in s.candidates:
        assert all(v == v for v in c.cv.secondary.values()), (c.id, c.cv.secondary)   # no NaN
    # the status here is random: the summary must say there is no signal instead of naming drivers
    assert s.results["test"]["vs_baseline"]["meaningful"] is False
    from workflows.report_graph import load_explain
    md = s.results["report_markdown"]
    assert "not more than chance" in md and "## Main drivers" not in md
    assert load_explain(s)["business"]["signal"] is False


def test_date_parts_are_scaled_with_numeric_columns():
    df = pd.DataFrame({"when": pd.to_datetime(["2026-01-01", "2026-03-05", "2026-07-09", "2026-11-30"] * 5),
                       "x": np.arange(20.0)})
    cols = {"numeric": ["x"], "categorical": [], "high_cardinality": [], "datetime": ["when"], "dropped": []}
    Xt = pc.build_preprocessor({"scale": True}, cols, True, 0).fit_transform(df)
    assert np.abs(np.asarray(Xt, dtype=float)).max() < 10   # no raw year (2026) left for saga to crawl over


def test_excluded_models_are_skipped_and_logged():
    from tools import sklearn_tools as sk
    plan = {"task_family": "classification", "target": "y", "metric": "f1_macro"}
    out = sk.candidate_specs(plan, {}, 2000, 5, True, None, ("gradient_boosting",))
    models = [s["model"] for s in out["specs"]]
    assert "gradient_boosting" not in models and len(models) >= 2
    assert any("EXCLUDE_MODELS" in r and "gradient_boosting" in r for r in out["conditioning"])
    everything = tuple({s["model"] for s in sk.candidate_specs(plan, {}, 2000, 5, True)["specs"]})
    assert len(sk.candidate_specs(plan, {}, 2000, 5, True, None, everything)["specs"]) >= 2   # never left with nothing


def _summary(vs, score=0.2553, base=0.25):
    from agents.business.core import BusinessInsightAgent
    plan = TaskPlan(task_family="classification", target="status", metric="pr_auc", rationale="t", confidence=0.9)
    test = {"score": score, "baseline": {"score": base}, "vs_baseline": vs}
    importance = {"importance": {"total_cents": 0.02, "placed_at": 0.01, "customer_country": 0.005}}
    return BusinessInsightAgent(AgentContext.offline()).stakeholder_summary(plan, test, importance, None, None, {"final_test_score": 18})


def test_summary_names_no_drivers_when_the_gain_is_chance():
    b = _summary({"applicable": True, "gain": 0.0053, "ci95": [-0.021, 0.03], "meaningful": False, "n_rows": 400,
                  "method": "paired bootstrap over the held-out rows"})
    text = " ".join([b["headline"]] + [r["text"] for r in b["recommendations"]] + b["caveats"])
    assert b["signal"] is False
    assert "driven" not in text and "total_cents" not in text
    assert "no better than chance" in b["headline"] and "Do not use this model" in text
    assert "within what chance produces on 400 held-out rows" in " ".join(b["caveats"])


def test_summary_names_drivers_when_the_gain_is_real():
    b = _summary({"applicable": True, "gain": 0.21, "ci95": [0.15, 0.27], "meaningful": True, "n_rows": 400,
                  "method": "paired bootstrap over the held-out rows"}, score=0.46)
    assert b["signal"] is True
    assert any("driven mostly by total_cents" in r["text"] for r in b["recommendations"])


def test_learnable_target_names_its_drivers(tmp_path):
    rng = np.random.default_rng(5)
    n = 400
    df = pd.DataFrame({"tenure": rng.integers(1, 60, n), "noise": rng.normal(size=n), "region": rng.choice(["n", "s"], n)})
    df["churn"] = np.where(df["tenure"] + rng.normal(0, 4, n) < 20, "yes", "no")
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    s = PipelineRun(store.create(str(path), "Predict churn", flags=RunFlags(memory=False, tuning=False)).id,
                    interactive=False).run()     # explanations on: the drivers come from the importance analysis
    assert s.status == "completed", s.error
    assert s.results["test"]["vs_baseline"]["meaningful"] is True
    from workflows.report_graph import load_explain
    business = load_explain(s)["business"]
    assert business["signal"] is True and any("driven mostly by tenure" in r["text"] for r in business["recommendations"])
    assert "## Main drivers" in s.results["report_markdown"]
