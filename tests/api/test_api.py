import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.runner import get_runner, reset_runner


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000")
    from app.config import reset_settings
    reset_settings()
    reset_runner()
    import app.router as r
    r._rate.clear()
    from app.main import create_app
    with TestClient(create_app()) as c:
        yield c
    for sid in list(get_runner()._active):
        get_runner().wait(sid)
    reset_runner()


def _upload(client, path, statement, mode="autopilot", name=None):
    with open(path, "rb") as f:
        return client.post("/sessions", files={"file": (name or path.name, f, "text/csv")},
                           data={"problem_statement": statement, "mode": mode})


def _wait(client, sid, until=("completed", "failed", "awaiting_approval", "awaiting_clarification", "blocked"), timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"/sessions/{sid}").json()
        if s["status"] in until and not s["running"]:
            return s
        time.sleep(0.3)
    raise TimeoutError(s)


def _events(client, sid, after=0):
    out = []
    with client.stream("GET", f"/sessions/{sid}/events?after={after}") as r:
        for line in r.iter_lines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:]))
    return out


FAST = {"tuning": False, "memory": False}


def test_autopilot_end_to_end(client, fixtures_dir):
    r = _upload(client, fixtures_dir / "regression.csv", "Estimate the sale price of each house")
    assert r.status_code == 200
    sid, profile = r.json()["session_id"], r.json()["profile"]
    assert profile["rows"] == 200
    assert client.post(f"/sessions/{sid}/start", json={"flags": FAST}).json()["started"]
    s = _wait(client, sid)
    assert s["status"] == "completed", s["error"]
    evs = _events(client, sid)
    seqs = [e["seq"] for e in evs]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    # resume from a sequence number: nothing lost, nothing repeated
    tail = _events(client, sid, after=seqs[10])
    assert [e["seq"] for e in tail] == seqs[11:]
    res = client.get(f"/sessions/{sid}/results").json()
    assert res["test"]["score"] > 0 and res["plots"] and res["ledger"]
    z = client.get(f"/sessions/{sid}/export")
    assert z.status_code == 200 and z.headers["content-type"] == "application/zip"
    assert client.get(f"/sessions/{sid}/export?file=report.html").status_code == 200
    assert client.get(f"/sessions/{sid}/export?file=../../etc/passwd").status_code == 404
    a = client.post(f"/sessions/{sid}/chat", json={"question": "why was this model chosen?"}).json()
    assert a["answer"] and a["cited"]


def test_review_mode_pauses_and_resumes(client, fixtures_dir):
    sid = _upload(client, fixtures_dir / "classification.csv", "Predict whether a customer will churn",
                  mode="review").json()["session_id"]
    client.post(f"/sessions/{sid}/start", json={"flags": FAST})
    seen = []
    for _ in range(6):
        s = _wait(client, sid)
        if s["status"] == "completed":
            break
        assert s["status"] == "awaiting_approval"
        seen.append(s["pending"]["key"])
        body = {"decision": "approve"}
        assert client.post(f"/sessions/{sid}/approve", json=body).status_code == 200
    assert s["status"] == "completed", s
    assert seen[:3] == ["preprocessing", "feature_engineering", "modelling"]
    types = {e.get("key") for e in _events(client, sid) if "key" in e}
    assert {"preprocessing", "feature_engineering", "modelling"} <= types


def test_override_warns_once_and_logs(client, fixtures_dir):
    sid = _upload(client, fixtures_dir / "regression.csv", "Estimate the sale price of each house",
                  mode="review").json()["session_id"]
    client.post(f"/sessions/{sid}/start", json={"flags": FAST})
    while True:
        s = _wait(client, sid)
        if s["pending"]["key"] == "modelling":
            break
        client.post(f"/sessions/{sid}/approve", json={"decision": "approve"})
    cands = s["pending"]["payload"]["candidates"]
    worst = sorted(cands, key=lambda c: -c["cv_mean"])[0]["id"]      # rmse: highest is worst
    r = client.post(f"/sessions/{sid}/approve", json={"decision": "change", "choice": worst, "justification": "simpler to run"})
    assert r.status_code == 200 and r.json()["warning"] and "underperform" in r.json()["warning"]
    s = _wait(client, sid)
    assert s["status"] == "completed" and s["chosen_id"].startswith(worst.split("_r")[0].split("_tuned")[0])
    res = client.get(f"/sessions/{sid}/results").json()
    ov = [e for e in res["ledger"] if e["decision_point"] == "model_choice_override"]
    assert len(ov) == 1 and ov[0]["overridden"] and ov[0]["decided_by"] == "client"
    assert res["test"]["shadow"]["recommended_score"] is not None
    warnings = [e for e in _events(client, sid) if "message" in e and "underperform" in e.get("message", "")]
    assert len(warnings) == 1


def test_clarification_flow(client, fixtures_dir):
    sid = _upload(client, fixtures_dir / "two_targets.csv", "Look at price and rent for these homes").json()["session_id"]
    client.post(f"/sessions/{sid}/start", json={"flags": FAST})
    s = _wait(client, sid)
    assert s["status"] == "awaiting_clarification"
    assert client.post(f"/sessions/{sid}/clarify", json={"choice": "nonsense"}).status_code == 422
    assert client.post(f"/sessions/{sid}/clarify", json={"choice": "rent"}).status_code == 200
    s = _wait(client, sid)
    # price predicts rent almost perfectly in this fixture: the auditor rightly stops on a leakage blocker
    assert s["status"] == "blocked" and s["pending"]["payload"]["blockers"] == ["price"]
    client.post(f"/sessions/{sid}/approve", json={"blockers": {"price": {
        "decision": "keep", "explanation": "the listing price is known before the rent is set"}}})
    s = _wait(client, sid, until=("completed", "failed"))
    assert s["status"] == "completed" and s["task_plan"]["target"] == "rent"


def test_leakage_blocker_pauses_run(client, fixtures_dir, tmp_path):
    df = pd.read_csv(fixtures_dir / "classification.csv")
    df["outcome_flag"] = df["churn"] + np.random.default_rng(0).normal(0, 0.01, len(df))
    p = tmp_path / "leaky.csv"
    df.to_csv(p, index=False)
    sid = _upload(client, p, "Predict whether a customer will churn").json()["session_id"]
    client.post(f"/sessions/{sid}/start", json={"flags": FAST})
    s = _wait(client, sid)
    assert s["status"] == "blocked" and s["pending"]["payload"]["blockers"] == ["outcome_flag"]
    from app.events import events_after   # the stream stays open while paused, so read the log directly
    blocker_events = [e for e in events_after(sid) if e["type"] == "blocker"]
    assert blocker_events
    r = client.post(f"/sessions/{sid}/approve", json={"blockers": {"outcome_flag": {"decision": "keep"}}})
    assert r.status_code == 422          # keeping a leak needs an explanation
    client.post(f"/sessions/{sid}/approve", json={"blockers": {"outcome_flag": {"decision": "drop"}}})
    s = _wait(client, sid, until=("completed", "failed"))
    assert s["status"] == "completed"
    res = client.get(f"/sessions/{sid}/results").json()
    assert "outcome_flag" not in res["clean"]["columns"]


def test_drift_endpoint(client, fixtures_dir, tmp_path):
    sid = _upload(client, fixtures_dir / "regression.csv", "Estimate the sale price of each house").json()["session_id"]
    client.post(f"/sessions/{sid}/start", json={"flags": FAST})
    _wait(client, sid)
    df = pd.read_csv(fixtures_dir / "regression.csv")
    df["sqft"] = df["sqft"] * 1.8
    p = tmp_path / "batch.csv"
    df.to_csv(p, index=False)
    with open(p, "rb") as f:
        r = client.post(f"/sessions/{sid}/drift", files={"file": ("batch.csv", f, "text/csv")}).json()
    assert "sqft" in r["drifted"] and r["limits"]


def test_upload_rules(client, fixtures_dir, tmp_path, monkeypatch):
    p = tmp_path / "x.txt"
    p.write_text("a,b\n1,2\n")
    assert _upload(client, p, "predict b").status_code == 415
    empty = tmp_path / "e.csv"
    empty.write_text("")
    assert _upload(client, empty, "predict b").status_code == 400
    header = tmp_path / "h.csv"
    header.write_text("a,b\n")
    r = _upload(client, header, "predict b")
    assert r.status_code == 422 and "no data rows" in r.json()["detail"]
    r = _upload(client, fixtures_dir / "classification.csv", "predict churn", name="../../etc/passwd.csv")
    sid = r.json()["session_id"]
    from app.session import store
    path = store.get(sid).file_path
    assert sid in path and "etc" not in path.split(sid)[1].replace("etc_passwd", "")
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")
    from app.config import reset_settings
    reset_settings()
    assert _upload(client, fixtures_dir / "classification.csv", "predict churn").status_code == 413


def test_unknown_session_and_cors(client):
    assert client.get("/sessions/notarealid").status_code == 404
    assert client.get("/sessions/" + "0" * 32).status_code == 404
    r = client.options("/healthz", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") != "http://evil.example"
    r = client.options("/healthz", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert client.get("/healthz").json() == {"ok": True}
    assert client.get("/readyz").json()["db"] is True


def test_rate_limit(client, fixtures_dir):
    import app.router as r
    r.RATE_LIMIT = (2, 3600)
    try:
        codes = [_upload(client, fixtures_dir / "classification.csv", "predict churn").status_code for _ in range(3)]
    finally:
        r.RATE_LIMIT = (20, 3600)
    assert codes == [200, 200, 429]
