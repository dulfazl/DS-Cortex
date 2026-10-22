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
