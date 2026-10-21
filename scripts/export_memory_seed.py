"""Export the memory store (fingerprints and pipelines, never raw data) to memory/seed_fingerprints.json,
which the API loads into an empty store at startup.

    python -m scripts.export_memory_seed reports/eval/cache/state/eval.sqlite --clean-dir reports/eval/cache/datasets

With --clean-dir, only runs on those CSV files are kept (flaw-injected variants are left out).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db", nargs="?", help="SQLite file to export from (default: the app database)")
    ap.add_argument("--clean-dir", help="keep only fingerprints of the CSV files in this folder")
    args = ap.parse_args()
    if args.db:
        os.environ["DB_PATH"] = args.db
    from app.db import init_db
    from memory.fingerprint_store import SEED_FILE, dataset_id, export_seed
    init_db()
    n = export_seed()
    if args.clean_dir:
        keep_ids = {dataset_id(str(p)) for p in Path(args.clean_dir).glob("*.csv")}
        rows = [r for r in json.loads(SEED_FILE.read_text()) if r["dataset_id"] in keep_ids]
        SEED_FILE.write_text(json.dumps(rows, indent=1))
        n = len(rows)
    print(f"exported {n} fingerprints to {SEED_FILE}")


if __name__ == "__main__":
    main()
