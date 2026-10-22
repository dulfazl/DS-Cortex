"""Central settings. Everything is read from the environment (or .env) so tests,
Docker and the eval harness can override paths without touching code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SEED = int(os.getenv("SEED", "42"))


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


@dataclass(frozen=True)
class Settings:
    google_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_fallbacks: tuple[str, ...] = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
    llm_mode: str = "auto"  # auto | gemini | offline
    seed: int = SEED
    data_dir: Path = ROOT / "datasets"
    log_dir: Path = ROOT / "logs"
    db_path: Path = ROOT / "logs" / "ds_cortex.sqlite"
    reports_dir: Path = ROOT / "reports"
    max_upload_mb: int = 50
    max_profile_rows: int = 200_000
    max_concurrent_runs: int = 2
    data_retention_hours: int = 72
    daily_call_budget: int = 1000
    allowed_origins: tuple[str, ...] = ("http://localhost:3000",)
    allowed_origin_regex: str | None = None
    cv_folds: int = 5
    cv_repeats: int = 3
    fast: bool = False  # fewer repeats / tuning trials (eval harness, tests)
    blocker_policy: str = "ask"  # ask | drop | keep  (non-interactive runs use drop)
    static_plots: bool = True    # PNG copies of plots in the bundle (needs Chrome for Kaleido)
    max_rows: int = 0            # 0 = no cap; small hosts sample larger files down (logged in the ledger)
    extra: dict = field(default_factory=dict)

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = _path("DATA_DIR", ROOT / "datasets")
    log_dir = _path("LOG_DIR", ROOT / "logs")
    origins = tuple(o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip())
    s = Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        gemini_fallbacks=tuple(m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",")
                               if m.strip()),
        llm_mode=os.getenv("LLM_MODE", "auto"),
        seed=int(os.getenv("SEED", "42")),
        data_dir=data_dir,
        log_dir=log_dir,
        db_path=_path("DB_PATH", log_dir / "ds_cortex.sqlite"),
        reports_dir=_path("REPORTS_DIR", ROOT / "reports"),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
        max_profile_rows=int(os.getenv("MAX_PROFILE_ROWS", "200000")),
        max_concurrent_runs=int(os.getenv("MAX_CONCURRENT_RUNS", "2")),
        data_retention_hours=int(os.getenv("DATA_RETENTION_HOURS", "72")),
        daily_call_budget=int(os.getenv("DAILY_CALL_BUDGET", "1000")),
        allowed_origins=origins,
        allowed_origin_regex=os.getenv("ALLOWED_ORIGIN_REGEX") or None,
        cv_folds=int(os.getenv("CV_FOLDS", "5")),
        cv_repeats=int(os.getenv("CV_REPEATS", "3")),
        fast=os.getenv("FAST_MODE", "0") == "1",
        blocker_policy=os.getenv("BLOCKER_POLICY", "ask"),
        static_plots=os.getenv("STATIC_PLOTS", "1") == "1",
        max_rows=int(os.getenv("MAX_ROWS", "0")),
    )
    for d in (s.data_dir, s.log_dir, s.upload_dir, s.sessions_dir, s.db_path.parent):
        d.mkdir(parents=True, exist_ok=True)
    return s


def reset_settings() -> None:
    """Tests change env vars and call this to rebuild settings."""
    get_settings.cache_clear()


def package_versions() -> dict[str, str]:
    """Logged with every run so results are reproducible."""
    import importlib.metadata as md
    import platform

    names = ["pandas", "numpy", "scikit-learn", "scipy", "statsmodels", "xgboost", "lightgbm",
             "shap", "plotly", "optuna", "pydantic", "fastapi", "google-genai", "google-adk"]
    out = {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}"}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            pass
    return out
