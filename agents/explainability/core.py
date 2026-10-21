"""Explainability: narrates decisions, not only predictions. Every sentence is drawn from the
Decision Ledger; two depths, beginner and technical."""
from __future__ import annotations

from agents.base import BaseAgent
from app.llm import narrate
from app.schemas import LedgerEntry

KEY_POINTS = ("task_family", "primary_metric", "apply_cleaning_plan", "holdout_split", "statistical_guidance",
              "memory_lookup", "candidate_generation", "model_choice", "model_choice_override", "final_pipeline",
              "final_test_score")


def _template(e: LedgerEntry, depth: str) -> str:
    r = e.recommendation
    dp = e.decision_point
    if depth == "beginner":
        lead = {
            "task_family": "First we worked out what kind of problem this is",
            "primary_metric": "We fixed how success is measured before trying any model",
            "apply_cleaning_plan": "We cleaned the data using only fixes that cannot leak information",
            "holdout_split": "We locked away part of the data to test the final model fairly",
            "statistical_guidance": "We checked the statistics behind the data",
            "memory_lookup": "We looked for similar datasets we had seen before",
            "candidate_generation": "We tried several kinds of model",
            "model_choice": "The agents debated which model to use",
            "model_choice_override": "You chose a different model than we recommended",
            "final_pipeline": "We fine-tuned the best options",
            "final_test_score": "Finally we tested the chosen model once on the locked-away data",
        }.get(dp, dp.replace("_", " ").capitalize())
        return f"{lead}: {r.recommendation}."
    conf = f" (confidence {r.confidence:.2f}, {r.confidence_method})"
    who = f" Decided by {e.decided_by}" + (f"; the client chose {e.client_choice}" if e.client_choice else "") + "."
    return f"{dp}: {r.recommendation}{conf}. {r.rationale}.{who}".replace("..", ".")


class ExplainabilityAgent(BaseAgent):
    name = "explainability"

    def narrate_decisions(self, entries: list[LedgerEntry], depth: str = "technical") -> dict:
        self.started(f"narrating decisions ({depth})")
        key = [e for e in entries if e.decision_point in KEY_POINTS or e.decision_point.startswith("blocker:")
               or e.decision_point.startswith("approve:")]
        items = [{"step": e.step, "decision_point": e.decision_point, "text": _template(e, depth)} for e in key]
        facts = {"decisions": [{"step": e.step, "decision": e.decision_point, "chosen": e.chosen,
                                "recommendation": e.recommendation.recommendation, "rationale": e.recommendation.rationale,
                                "confidence": e.recommendation.confidence, "decided_by": e.decided_by,
                                "overridden": e.overridden} for e in key]}
        fallback = " ".join(i["text"] for i in items)
        summary, source = narrate(self.ctx.llm, "Summarize why each decision in this data-science run was made, in order. "
                                                "Refer to steps by number.", facts, fallback, depth=depth)
        self.finished(f"{len(items)} decisions narrated ({source})")
        return {"depth": depth, "items": items, "summary": summary, "source": source}
