"""FastAPI app: CORS, routes, startup recovery, data retention and health checks.

Run with ONE worker: event queues live in SQLite but run threads and their pauses belong to the
process that started them. `uvicorn app.main:app --workers 1`."""
from __future__ import annotations

import logging
import shutil
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import db, events
from app.config import get_settings
from app.replay import replay_api
from app.router import api
from app.session import store

log = logging.getLogger("ds_cortex")


def recover() -> None:
    """A restart interrupts running threads: mark them so the client can resume them with /start.
    Runs waiting for an answer keep their pending question and resume when it is answered."""
    for row in store.list():
        if row["status"] in ("running", "queued"):
            s = store.get(row["id"])
            s.status = "interrupted"
            store.save(s)
            events.emit(s.id, "status", {"status": "interrupted", "message": "the server restarted; press Resume to continue"})


def purge_old() -> int:
    s = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=s.data_retention_hours)
    n = 0
    for row in store.list():
        try:
            created = datetime.fromisoformat(row["created_at"])
        except (TypeError, ValueError):
            continue
        if created < cutoff:
            sess = store.get(row["id"])
            if sess:
                shutil.rmtree(s.upload_dir / sess.id, ignore_errors=True)
            store.delete(row["id"])
            n += 1
    return n


def _retention_loop() -> None:
    while True:
        try:
            purge_old()
        except Exception:  # noqa: BLE001
            log.exception("retention sweep failed")
        time.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    recover()
    threading.Thread(target=_retention_loop, daemon=True, name="retention").start()
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="DS Cortex", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(s.allowed_origins), allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    app.include_router(api)
    app.include_router(replay_api)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/readyz")
    def readyz():
        try:
            with db.connect() as c:
                c.execute("SELECT 1").fetchone()
            ok_db = True
        except Exception:  # noqa: BLE001
            ok_db = False
        st = get_settings()
        body = {"db": ok_db, "llm_key": bool(st.google_api_key), "llm_mode": st.llm_mode}
        return JSONResponse(body, status_code=200 if ok_db else 503)

    return app


app = create_app()
