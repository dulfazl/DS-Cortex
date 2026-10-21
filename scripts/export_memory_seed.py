"""Export the memory store (fingerprints and pipelines, never raw data) to memory/seed_fingerprints.json,
which the API loads into an empty store at startup.

    python -m scripts.export_memory_seed [path/to/state.sqlite]
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    if len(sys.argv) > 1:
        os.environ["DB_PATH"] = sys.argv[1]
    from app.db import init_db
    from memory.fingerprint_store import SEED_FILE, export_seed
    init_db()
    print(f"exported {export_seed()} fingerprints to {SEED_FILE}")


if __name__ == "__main__":
    main()
