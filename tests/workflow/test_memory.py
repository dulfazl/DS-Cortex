"""Phase 10 checkpoint: a similar dataset gets the suggestion (as a candidate, shown as advice);
a mismatched suggestion is rejected by the Critic and the rejection is logged."""
import json

import numpy as np
import pandas as pd

from app.ledger import Ledger
from app.session import RunFlags, store
from memory import fingerprint_store as fs
from workflows.pipeline_graph import PipelineRun

FLAGS = RunFlags(tuning=False, explain=False)


def _similar_copy(fixtures_dir, tmp_path, seed):
    df = pd.read_csv(fixtures_dir / "classification.csv").sample(frac=0.9, random_state=seed)
    p = tmp_path / f"similar_{seed}.csv"
    df.to_csv(p, index=False)
    return p


def test_memory_suggests_on_similar_data(fixtures_dir, tmp_path):
    first = store.create(str(fixtures_dir / "classification.csv"), "Predict whether a customer will churn", flags=FLAGS)
    first = PipelineRun(first.id, interactive=False).run()
    assert first.status == "completed"
    assert fs.memory_stats()["fingerprints_stored"] == 1

    second = store.create(str(_similar_copy(fixtures_dir, tmp_path, 1)), "Predict whether a customer will churn", flags=FLAGS)
    second = PipelineRun(second.id, interactive=False).run()
    assert second.status == "completed", second.error
    sugg = second.memory["suggestions"]
    assert sugg and sugg[0]["distance"] <= fs.DISTANCE_THRESHOLD
    mem = [c for c in second.candidates if c.origin == "memory"]
    assert mem and mem[0].objections is not None            # it entered the candidates and was reviewed
    lookup = next(e for e in Ledger(second.id).entries() if e.decision_point == "memory_lookup")
    assert "advice only" in lookup.chosen
    assert second.memory["outcome"]["suggested"] >= 1 and second.memory["outcome"]["entered"] >= 1


def test_mismatched_suggestion_is_rejected_and_logged(fixtures_dir, tmp_path):
    # a stored run with the same shape of data but a different target type (multiclass vs binary)
    probe = store.create(str(fixtures_dir / "classification.csv"), "Predict whether a customer will churn",
                         flags=RunFlags(memory=False, tuning=False, explain=False))
    probe = PipelineRun(probe.id, interactive=False).run()
    fs.save("other_dataset", "classification", probe.stats_profile.fingerprint,
            {"model": "random_forest", "params": {"n_estimators": 50}, "prep": {}, "balance": None},
            "random_forest", 0.9, 0.3, "f1_macro", "seeded", "multiclass", seeded=True)
    s = store.create(str(_similar_copy(fixtures_dir, tmp_path, 2)), "Predict whether a customer will churn", flags=FLAGS)
    s = PipelineRun(s.id, interactive=False).run()
    assert s.status == "completed", s.error
    mem = [c for c in s.candidates if c.origin == "memory" and c.model == "random_forest"]
    assert mem and mem[0].status == "rejected"
    assert not any(c.parent == mem[0].id for c in s.candidates)          # a rejected analogy is never revised
    assert any(o.check == "memory_fit_check" and "multiclass" in o.message for o in mem[0].objections)
    rejections = [e for e in Ledger(s.id).entries() if e.decision_point == f"reject:{mem[0].id}"]
    assert rejections and "multiclass" in rejections[0].recommendation.rationale


def test_far_fingerprint_gives_no_suggestion():
    fs.save("far", "classification", [1.0] * 12, {"model": "logreg", "params": {}, "prep": {}}, "logreg", 0.8, 0.5,
            "f1_macro", "x", "binary")
    res = fs.find_similar([0.0] * 12, "classification")
    assert res["matches"] == [] and res["nearest_distance"] > fs.DISTANCE_THRESHOLD
    assert fs.find_similar([1.0] * 12, "regression")["matches"] == []     # other task families never match
