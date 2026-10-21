import pytest

from app.ledger import Ledger
from app.schemas import Evidence, EvidenceError, Recommendation, validate_recommendation
from app.session import store
from tools.tool_log import ToolLog


def _rec(evidence, method="rule", point="imputation"):
    return Recommendation(agent="data_auditor", decision_point=point, recommendation="impute median",
                          rationale="skewed column", evidence=evidence, confidence=0.8, confidence_method=method)


def _ev():
    return Evidence(tool="pandas_tools.missing_report", inputs={"col": "x"}, result={"missing_pct": 12.0}, run_id="r1")


def test_empty_evidence_raises():
    with pytest.raises(EvidenceError):
        validate_recommendation(_rec([]))


def test_llm_confidence_rejected_when_computed_available():
    with pytest.raises(EvidenceError):
        validate_recommendation(_rec([_ev()], method="llm_estimate"))
    assert validate_recommendation(_rec([_ev()], method="llm_estimate"), computed_available=False)


def test_confidence_bounds():
    with pytest.raises(Exception):
        Recommendation(agent="a", decision_point="d", recommendation="r", rationale="", evidence=[_ev()],
                       confidence=1.5, confidence_method="rule")


def test_session_with_three_ledger_entries_reloads(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    s = store.create(str(csv), "predict b", mode="review")
    log = ToolLog(s.id)
    ledger = Ledger(s.id)
    for i in range(3):
        _, ev = log.run("pandas_tools.describe_csv", lambda: {"rows": 2, "i": i}, {"path": "x.csv"})
        ledger.add(_rec([ev], point=f"point_{i}"), options=["a", "b"], chosen="a")
    reloaded = store.get(s.id)
    assert reloaded.mode == "review" and reloaded.problem_statement == "predict b"
    entries = Ledger(s.id).entries()
    assert [e.step for e in entries] == [1, 2, 3]
    assert entries[2].recommendation.evidence[0].result["i"] == 2
    assert log.get(entries[0].recommendation.evidence[0].run_id)["result"]["rows"] == 2
    md = ledger.export("markdown")
    assert "point_1" in md and "Decision Ledger" in md


def test_ledger_rejects_unevidenced_entry(tmp_path):
    s = store.create(str(tmp_path / "x.csv"), "x")
    with pytest.raises(EvidenceError):
        Ledger(s.id).add(_rec([]), options=["a"], chosen="a")
    assert Ledger(s.id).entries() == []
