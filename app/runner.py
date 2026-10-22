"""Background runs. Each run is a thread (the tools are CPU-bound); a semaphore caps concurrent
runs, and later starts wait in a queue with a `queued` event. A pause ends the thread; answering
it starts a new thread that resumes from the saved stage."""
from __future__ import annotations

import threading
import time

from app import events
from app.config import get_settings
from app.session import store


class Runner:
    def __init__(self, max_concurrent: int | None = None):
        self.max = max_concurrent or get_settings().max_concurrent_runs
        self._sem = threading.BoundedSemaphore(self.max)
        self._lock = threading.Lock()
        self._active: dict[str, threading.Thread] = {}
        self._queue: list[str] = []
        self._restart: dict[str, dict] = {}

    def is_active(self, sid: str) -> bool:
        with self._lock:
            return sid in self._active

    def queue_position(self, sid: str) -> int | None:
        with self._lock:
            return self._queue.index(sid) + 1 if sid in self._queue else None

    def start(self, sid: str, **run_kwargs) -> bool:
        with self._lock:
            if sid in self._active:
                # the previous thread may be finishing after a pause that has just been answered:
                # remember the request and start again when it exits
                self._restart[sid] = run_kwargs
                return False
            t = threading.Thread(target=self._work, args=(sid, run_kwargs), daemon=True, name=f"run-{sid[:8]}")
            self._active[sid] = t
            self._queue.append(sid)
        t.start()
        return True

    def _work(self, sid: str, run_kwargs: dict) -> None:
        try:
            if not self._sem.acquire(blocking=False):
                s = store.get(sid)
                if s:
                    s.status = "queued"
                    store.save(s)
                events.emit(sid, "queued", {"position": self.queue_position(sid),
                                            "message": f"{self.max} run(s) already in progress; yours starts next in line"})
                self._sem.acquire()
            with self._lock:
                if sid in self._queue:
                    self._queue.remove(sid)
            from workflows.pipeline_graph import PipelineRun
            try:
                PipelineRun(sid, **run_kwargs).run()
            finally:
                self._sem.release()
        except Exception as e:  # noqa: BLE001 - never let a thread die silently
            events.emit(sid, "failed", {"message": f"run could not start: {e}"})
            s = store.get(sid)
            if s:
                s.status, s.error = "failed", str(e)
                store.save(s)
        finally:
            with self._lock:
                self._active.pop(sid, None)
                if sid in self._queue:
                    self._queue.remove(sid)
                again = self._restart.pop(sid, None)
            if again is not None:
                s = store.get(sid)
                if s and s.status not in ("completed", "failed") and not s.pending:
                    self.start(sid, **again)

    def wait(self, sid: str, timeout: float = 600) -> None:
        """Block until the run thread for sid ends (tests and scripts)."""
        t0 = time.time()
        while self.is_active(sid) and time.time() - t0 < timeout:
            time.sleep(0.1)


_runner: Runner | None = None


def get_runner() -> Runner:
    global _runner
    if _runner is None:
        _runner = Runner()
    return _runner


def reset_runner() -> None:
    global _runner
    _runner = None
