"""Export completed runs as static replays for the demo website, which then needs no server.

    python -m scripts.export_demo OUT_DIR SESSION_ID:NAME ...

Writes OUT_DIR/<name>.json (the run's events with their timing, and its results, in the format the replay
API serves) and OUT_DIR/index.json (one card per run). Pauses longer than a few seconds, such as a person
reading an approval bundle, are shortened so a replay plays in a few minutes."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from app import events
from app.db import init_db
from app.router import results
from app.session import store

MAX_GAP_SECONDS = 3.0


def export(sid: str, name: str, out: Path) -> dict:
    s = store.get(sid)
    if s is None or s.status != "completed":
        sys.exit(f"{sid}: no such session, or it has not completed")
    evs = events.events_after(sid, 0, limit=100000)
    offset, prev = 0.0, None
    for e in evs:
        ts = datetime.fromisoformat(e["ts"])
        if prev is not None:
            offset += min((ts - prev).total_seconds(), MAX_GAP_SECONDS)
        prev = ts
        e["offset"] = round(offset, 3)
    res = results(sid)
    (out / f"{name}.json").write_text(json.dumps({"name": name, "events": evs, "results": res}, separators=(",", ":")))
    test = res.get("test") or {}
    return {"name": name, "file_name": s.file_name, "problem_statement": s.problem_statement,
            "task": s.task_plan.task_family if s.task_plan else None, "metric": test.get("metric"),
            "score": test.get("score"), "baseline": (test.get("baseline") or {}).get("score"),
            "beyond_chance": (test.get("vs_baseline") or {}).get("meaningful"), "events": len(evs),
            "duration_seconds": round(offset)}


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    init_db()
    index = [export(*arg.split(":", 1), out) for arg in sys.argv[2:]]
    (out / "index.json").write_text(json.dumps(index, indent=1))
    for row in index:
        print(f"{row['name']}: {row['events']} events, plays in {row['duration_seconds']} s, {row['metric']} {row['score']}")


if __name__ == "__main__":
    main()
