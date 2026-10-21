"""Every test gets its own temp data dir and SQLite file, and runs offline (no Gemini)."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "logs" / "test.sqlite"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("LLM_MODE", os.getenv("TEST_LLM_MODE", "offline"))
    monkeypatch.setenv("FAST_MODE", "1")
    monkeypatch.setenv("CV_REPEATS", "1")
    from app.config import reset_settings
    reset_settings()
    from app.db import init_db
    init_db()
    yield
    reset_settings()


@pytest.fixture
def fixtures_dir():
    return FIXTURES
