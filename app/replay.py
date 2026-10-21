"""Demo safety net: a recorded "golden run" (every event with its timing, plus the results) that
can be streamed again at its original pace when the network or the Gemini quota fails during a
presentation. Replays are always labelled as replays."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app import events
from app.config import get_settings


def replay_dir() -> Path:
    return get_settings().reports_dir / "replays"


replay_api = APIRouter()


def record(session_id: str, name: str, results: dict) -> Path:
    folder = replay_dir()
    folder.mkdir(parents=True, exist_ok=True)
    evs = events.events_after(session_id, 0, limit=100000)
    t0 = datetime.fromisoformat(evs[0]["ts"]) if evs else None
    for e in evs:
        e["offset"] = (datetime.fromisoformat(e["ts"]) - t0).total_seconds() if t0 else 0
    path = folder / f"{name}.json"
    path.write_text(json.dumps({"name": name, "session_id": session_id, "events": evs, "results": results}))
    return path


def _load(name: str) -> dict:
    if not re.fullmatch(r"[\w\-]{1,60}", name):
        raise HTTPException(404, "replay not found")
    p = replay_dir() / f"{name}.json"
    if not p.exists():
        raise HTTPException(404, "replay not found")
    return json.loads(p.read_text())


@replay_api.get("/replays")
def list_replays():
    folder = replay_dir()
    return [p.stem for p in sorted(folder.glob("*.json"))] if folder.exists() else []


@replay_api.get("/replays/{name}/results")
def replay_results(name: str):
    data = _load(name)
    return {**data["results"], "replay": True, "replay_name": name}


@replay_api.get("/replays/{name}/events")
async def replay_events(name: str, request: Request, speed: float = 1.0):
    data = _load(name)
    speed = max(0.1, min(speed, 50.0))

    async def gen():
        last = 0.0
        for e in data["events"]:
            if await request.is_disconnected():
                break
            await asyncio.sleep(max(0.0, (e["offset"] - last) / speed))
            last = e["offset"]
            yield {"event": e["type"], "id": str(e["seq"]), "data": json.dumps({**e["data"], "seq": e["seq"], "replay": True})}

    return EventSourceResponse(gen(), ping=15)
