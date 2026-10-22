"""Live smoke test against Gemini (nightly or by hand; skipped without GOOGLE_API_KEY).
One small dataset, so it stays inside the free tier."""
import os

import pytest

# opt-in: each run spends about 8 of the free tier's ~20 daily requests per model
pytestmark = pytest.mark.skipif(not (os.getenv("GOOGLE_API_KEY") and os.getenv("RUN_LIVE") == "1"),
                                reason="needs GOOGLE_API_KEY and RUN_LIVE=1")


def test_one_real_run(fixtures_dir, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "gemini")
    from app.config import reset_settings
    reset_settings()
    from app.session import RunFlags, store
    from workflows.pipeline_graph import PipelineRun
    s = store.create(str(fixtures_dir / "regression.csv"), "Estimate the sale price of each house",
                     flags=RunFlags(memory=False, tuning=False))
    s = PipelineRun(s.id, interactive=False).run()
    assert s.status == "completed", s.error
    assert s.usage.get("calls", 0) > 0
    # numbers in LLM narration that could not be traced were rejected, never shown
    assert s.debate.verifier["total"] >= s.debate.verifier["verified"]
