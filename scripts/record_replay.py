"""Record a completed run as a replay for demos (12C safety net).

    python -m scripts.record_replay SESSION_ID NAME
"""
from __future__ import annotations

import sys

from app.db import init_db
from app.replay import record
from app.router import results
from app.session import store


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sid, name = sys.argv[1], sys.argv[2]
    init_db()
    s = store.get(sid)
    if s is None or s.status != "completed":
        sys.exit("that session does not exist or has not completed")
    path = record(sid, name, results(sid))
    print(f"recorded {path}")


if __name__ == "__main__":
    main()
