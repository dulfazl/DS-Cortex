"""Full chain from a Python call: ledger, report, export bundle, reproduction, SHAP fallback,
determinism, test-set guard and the evidence-trace check."""
import json
import re
import subprocess
import sys
import zipfile

import pytest

from app.chat import answer
from app.ledger import Ledger
from app.llm import OfflineLLM
from app.session import RunFlags, store
from tools.verify_tools import extract_numbers, flatten_numbers, number_supported
from workflows.pipeline_graph import PipelineRun, TestSetGuard

STATEMENT = "Predict whether a customer will churn and explain the drivers"


def _run(fixtures_dir, flags=None, name="classification.csv", statement=STATEMENT):
    s = store.create(str(fixtures_dir / name), statement, flags=flags or RunFlags(memory=False))
    return PipelineRun(s.id, interactive=False).run()


@pytest.fixture(scope="module")
def done(tmp_path_factory, fixtures_dir=None):
    return None


def test_full_chain_produces_verified_decision_and_bundle(fixtures_dir, tmp_path):
    s = _run(fixtures_dir)
    assert s.status == "completed", s.error
    entries = Ledger(s.id).entries()
    points = [e.decision_point for e in entries]
    for p in ("task_family", "primary_metric", "apply_cleaning_plan", "holdout_split", "candidate_generation",
              "model_choice", "final_pipeline", "final_test_score"):
        assert p in points
    final = next(e for e in entries if e.decision_point == "model_choice")
    assert final.recommendation.claims and all(c.verified for c in final.recommendation.claims)
    assert s.stage_outputs["test_reads"] == 1
    z = zipfile.ZipFile(s.results["export"]["zip"])
    names = set(z.namelist())
    for f in ("report.md", "report.html", "pipeline.py", "requirements.txt", "ledger.json", "ledger.md",
              "debate_transcript.md", "model/model_card.md", "model/input_schema.json", "manifest.json"):
        assert f in names
    manifest = json.loads(z.read("manifest.json"))
    import hashlib
    for f, h in manifest["files"].items():
        assert hashlib.sha256(z.read(f)).hexdigest() == h
    # the exported script reproduces the headline score
    z.extract("pipeline.py", tmp_path)
    r = subprocess.run([sys.executable, "-I", "pipeline.py", "--data", str(fixtures_dir / "classification.csv")],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300)
    out = json.loads(r.stdout)
    assert out["matches_report"], out
    # run-scoped chat answers from this run's ledger
    a = answer(s, "why was this metric chosen?", OfflineLLM())
    assert any(c.get("decision_point") == "primary_metric" for c in a["cited"])


def test_html_report_escapes_model_text(fixtures_dir):
    s = _run(fixtures_dir)
    html = open(s.results["export"]["dir"] + "/report.html").read()
    assert "<script>alert" not in html
    s2 = store.create(str(fixtures_dir / "classification.csv"), "Predict churn <script>alert(1)</script>",
                      flags=RunFlags(memory=False, tuning=False))
    s2 = PipelineRun(s2.id, interactive=False).run()
    html = open(s2.results["export"]["dir"] + "/report.html").read()
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_shap_fallback_is_logged(fixtures_dir):
    s = _run(fixtures_dir, RunFlags(memory=False, tuning=False, force_shap_failure=True))
    assert s.results["explain_summary"]["shap_substituted"]
    assert any(e.decision_point == "explainer_substitution" for e in Ledger(s.id).entries())
    md = s.results["report_markdown"]
    assert "SHAP was substituted" in md


def test_determinism_same_seed_same_numbers(fixtures_dir):
    a = _run(fixtures_dir, RunFlags(memory=False, tuning=False))
    b = _run(fixtures_dir, RunFlags(memory=False, tuning=False))
    assert a.chosen_id == b.chosen_id
    assert a.results["test"]["score"] == b.results["test"]["score"]
    ea = [(e.decision_point, e.chosen) for e in Ledger(a.id).entries()]
    eb = [(e.decision_point, e.chosen) for e in Ledger(b.id).entries()]
    assert ea == eb


def test_test_set_guard_refuses_second_read(fixtures_dir):
    s = store.create(str(fixtures_dir / "classification.csv"), STATEMENT)
    g = TestSetGuard(s)
    g.read("final")
    with pytest.raises(RuntimeError):
        g.read("peek")


def test_evidence_trace_no_invented_numbers(fixtures_dir):
    """Every number in the report traces to a logged tool result (settings are allowed)."""
    s = _run(fixtures_dir, RunFlags(memory=False, tuning=False))
    from tools.tool_log import ToolLog
    facts = set()
    import sqlite3
    from app import db
    with db.connect() as c:
        for row in c.execute("SELECT result FROM tool_runs WHERE session_id=?", (s.id,)):
            flatten_numbers(json.loads(row["result"]), facts)
    facts |= flatten_numbers(s.profile)
    for e in Ledger(s.id).entries():   # confidences recorded in the ledger are computed by code, not stated by the LLM
        facts |= flatten_numbers(e.recommendation.model_dump(include={"confidence", "recommendation", "rationale"}))
    allowed = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 42.0, 50.0, 60.0, 80.0, 95.0, 100.0, 120.0}
    md = s.results["report_markdown"]
    md = re.sub(r"`[^`]*`", "", md)                       # file names and code
    md = re.sub(r"\b(c|m)\d+_[\w]+", "", md)               # candidate ids
    md = re.sub(r"step \d+|Step \d+|round \d+|^\d+\. ", "", md, flags=re.M)
    unmatched = [t for t in extract_numbers(md) if not number_supported(t, facts, allowed)]
    assert not unmatched, unmatched[:20]
