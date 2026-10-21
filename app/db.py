"""SQLite persistence. One short-lived connection per operation keeps it thread-safe;
WAL mode lets the SSE reader and the run thread work at the same time."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import get_settings

_write_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, data TEXT NOT NULL, status TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, step INTEGER NOT NULL,
    data TEXT NOT NULL, created_at TEXT);
CREATE INDEX IF NOT EXISTS ledger_session ON ledger(session_id, step);
CREATE TABLE IF NOT EXISTS tool_runs (
    run_id TEXT PRIMARY KEY, session_id TEXT, tool TEXT, inputs TEXT, result TEXT, created_at TEXT);
CREATE INDEX IF NOT EXISTS tool_runs_session ON tool_runs(session_id);
CREATE TABLE IF NOT EXISTS events (
    session_id TEXT NOT NULL, seq INTEGER NOT NULL, type TEXT NOT NULL, data TEXT, created_at TEXT,
    PRIMARY KEY (session_id, seq));
CREATE TABLE IF NOT EXISTS fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id TEXT, created_at TEXT, task_family TEXT,
    vector TEXT, pipeline_spec TEXT, model TEXT, test_score REAL, baseline_score REAL, metric TEXT,
    ledger_ref TEXT, target_type TEXT, seeded INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS memory_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, suggested INTEGER, entered INTEGER,
    won INTEGER, rejected INTEGER, score_diff REAL, distance REAL, data TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS usage (
    day TEXT PRIMARY KEY, calls INTEGER DEFAULT 0, tokens INTEGER DEFAULT 0);
"""


def _connect() -> sqlite3.Connection:
    s = get_settings()
    conn = sqlite3.connect(s.db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


@contextmanager
def connect(write: bool = False) -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        if write:
            with _write_lock:
                yield conn
                conn.commit()
        else:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with connect(write=True) as c:
        c.executescript(SCHEMA)


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)
