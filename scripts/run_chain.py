"""Run the full chain on one CSV from the command line (Phase 6 checkpoint).

    python -m scripts.run_chain tests/fixtures/classification.csv "Predict whether a customer will churn"
"""
from __future__ import annotations

import argparse
import json
import time

from app.db import init_db
from app.ledger import Ledger
from app.session import RunFlags, store
from workflows.pipeline_graph import PipelineRun


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("statement")
    ap.add_argument("--mode", default="autopilot", choices=["autopilot", "review"])
    ap.add_argument("--stop-after", default=None)
    ap.add_argument("--no-memory", action="store_true")
    args = ap.parse_args()
    init_db()
    s = store.create(args.csv, args.statement, mode=args.mode, flags=RunFlags(memory=not args.no_memory))
    t0 = time.time()
    run = PipelineRun(s.id, interactive=False,
                      on_event=lambda t, d: print(f"  [{t}] {d.get('agent') or d.get('stage') or d.get('decision_point') or ''}")
                      if t in ("agent_finished", "stage", "ledger_entry", "failed") else None)
    s = run.run(stop_after=args.stop_after)
    print(f"\nstatus={s.status} chosen={s.chosen_id} in {time.time() - t0:.1f}s")
    if s.error:
        print("error:", s.error)
    if s.debate:
        print("debate rule:", s.debate.rule, "| forced tiebreak:", s.debate.forced_tiebreak,
              "| verifier:", {k: v for k, v in s.debate.verifier.items() if k != "confidence_components"})
    print("test:", json.dumps(s.results.get("test"), default=str)[:400])
    entries = Ledger(s.id).entries()
    print(f"ledger: {len(entries)} entries")
    final = next((e for e in entries if e.decision_point == "model_choice"), None)
    if final:
        unverified = [c.text for c in final.recommendation.claims if c.verified is False]
        print("final decision unverified claims:", unverified)
    print("session:", s.id)


if __name__ == "__main__":
    main()
