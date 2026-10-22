"""An answer that lands while the paused run's thread is still finishing must not be lost."""
import threading
import time

from app.runner import Runner
from app.session import store
from workflows.pipeline_graph import Pause, PipelineRun


def test_final_save_keeps_an_answer_saved_meanwhile(fixtures_dir, monkeypatch):
    s = store.create(str(fixtures_dir / "classification.csv"), "Predict churn", mode="review")

    def pause_after_client_answers(self):
        # the client answers between the pause and the thread's last write
        other = store.get(self.s.id)
        other.answers["preprocessing"] = {"decision": "approve"}
        other.pending, other.status = None, "running"
        store.save(other)
        raise Pause("approval", "preprocessing", {})

    monkeypatch.setattr(PipelineRun, "stage_profile", pause_after_client_answers)
    PipelineRun(s.id, interactive=True).run()
    after = store.get(s.id)
    assert after.answers.get("preprocessing") == {"decision": "approve"}


def test_start_during_a_finishing_thread_runs_again(fixtures_dir, monkeypatch):
    s = store.create(str(fixtures_dir / "classification.csv"), "Predict churn")
    runs = []
    gate = threading.Event()

    def fake_run(self, stop_after=None):
        runs.append(time.time())
        if len(runs) == 1:
            gate.wait(5)            # the first thread is still finishing when the second start arrives
        return self.s

    monkeypatch.setattr(PipelineRun, "run", fake_run)
    r = Runner(max_concurrent=1)
    assert r.start(s.id) is True
    time.sleep(0.2)
    assert r.start(s.id) is False   # refused for now, but remembered
    gate.set()
    for _ in range(50):
        if len(runs) == 2 and not r.is_active(s.id):
            break
        time.sleep(0.1)
    assert len(runs) == 2
