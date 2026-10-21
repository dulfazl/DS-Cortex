"""Task Router: from the CSV and the problem statement, decide task family, target and metric.
Rules decide; the LLM may only map wording to an existing column. Below 0.7 confidence, or
with an unclear target, it asks one clarifying question instead of guessing."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from agents.base import BaseAgent
from app.llm import ask_json
from app.schemas import ClarifyingQuestion, TaskPlan
from tools import router_tools as rt

FAMILIES = ["classification", "regression", "clustering", "anomaly_detection", "time_series", "offline_rl"]
CLARIFY_THRESHOLD = 0.7


class RouterAgent(BaseAgent):
    name = "router"

    def plan(self, statement: str, profile: dict, df: pd.DataFrame,
             answer: dict[str, Any] | None = None) -> TaskPlan | ClarifyingQuestion:
        self.started("reading the problem statement and quick profile")
        cols = [c["name"] for c in profile["column_info"]]
        pre, ev = self.ctx.tool("router_tools.precheck", rt.precheck,
                                {"statement": statement, "columns": cols}, statement, profile, df)
        family, conf, target = pre["suggested_family"], pre["rule_confidence"], pre["target"]
        notes: list[str] = list(pre["reasons"])
        answer = answer or {}

        # --- client answers override the rules
        if answer.get("rl_mismatch") == "accept_supervised" and not answer.get("target"):
            t = pre["target"] or (pre["hint_targets"] or [None])[0]
            if t:
                answer = {**answer, "target": t}
        if answer.get("target") and answer["target"] in cols:
            target = answer["target"]
            pre, ev = self.ctx.tool("router_tools.precheck", rt.precheck,
                                    {"statement": statement, "columns": cols, "client_answer": answer},
                                    f"target is {target} . " + statement, profile, df)
            family, conf = pre["suggested_family"], 0.95
            notes.append(f"client named the target '{target}'")
        if answer.get("task_family") in FAMILIES:
            family, conf = answer["task_family"], 0.95
            if family in ("clustering",) or (family == "anomaly_detection" and not answer.get("target")):
                target = None
            notes.append(f"client chose the task family '{family}'")

        # --- RL honesty check
        if "rl" in pre["keyword_hits"] and not pre["rl_data"] and not answer:
            framing = rt.closest_supervised_framing(profile, pre)
            q = ClarifyingQuestion(
                question=("This data does not look like logged decisions: there is no action column and reward "
                          f"column, so offline RL does not fit. {framing} Continue with that framing?"),
                options=["Yes, use the supervised framing", "No, I will upload logged-decision data"],
                reason="rl_mismatch", kind="other")
            self.record("task_family", "ask: data does not fit RL", q.question, [ev], 1.0, "rule",
                        options=FAMILIES, chosen="clarify")
            self.finished("asked a clarifying question (RL mismatch)")
            return q

        # --- the LLM may map wording to a column, never invent one
        if not family and not target and self.ctx.llm.available:
            mapped = self._llm_map(statement, profile, pre)
            if mapped:
                pre, ev = self.ctx.tool("router_tools.precheck", rt.precheck,
                                        {"statement": statement, "llm_mapped_target": mapped, "columns": cols},
                                        f"target is {mapped}. " + statement, profile, df)
                family, target = pre["suggested_family"], pre["target"]
                conf = min(pre["rule_confidence"], 0.75)
                notes.append(f"the LLM mapped the wording to column '{mapped}' (verified to exist)")

        ambiguous = pre["ambiguous_targets"] and not answer.get("target")
        if family is None or conf < CLARIFY_THRESHOLD or ambiguous:
            q = self._question(pre, profile, ambiguous)
            self.record("task_family", "ask a clarifying question", q.reason, [ev], 1.0, "rule",
                        options=FAMILIES, chosen="clarify",
                        note=f"rule confidence {conf:.2f} is below {CLARIFY_THRESHOLD}" if conf < CLARIFY_THRESHOLD else None)
            self.finished("asked a clarifying question")
            return q

        tinfo = pre.get("target_info") or {}
        imbalanced = bool(family == "classification" and tinfo and
                          (tinfo.get("minority_share", 1) < 0.2 or tinfo.get("imbalance_ratio", 1) > 5))
        metrics = rt.metric_for(family, imbalanced=imbalanced, labeled=bool(target))
        plan = TaskPlan(
            task_family=family, target=target if family != "clustering" else None,
            metric=metrics["primary"], secondary_metrics=metrics["secondary"],
            rationale="; ".join(notes) or "rule-based routing",
            confidence=round(conf, 3), confidence_method="rule", imbalanced=imbalanced,
            time_column=pre["datetime_columns"][0] if family == "time_series" else None,
            group_column=self._group_column(profile, target),
            action_column=pre["rl_columns"]["action"] if family == "offline_rl" else None,
            reward_column=pre["rl_columns"]["reward"] if family == "offline_rl" else None,
            propensity_column=pre["rl_columns"]["propensity"] if family == "offline_rl" else None,
            explainability_required=pre["explainability_requested"],
            notes=notes, evidence=[ev])
        self.record("task_family", f"{family} (target: {plan.target or 'none'})", plan.rationale, [ev],
                    plan.confidence, "rule", options=FAMILIES, chosen=family)
        self.record("primary_metric", plan.metric,
                    f"fixed by the task-to-metric table for {family}" + (" (imbalanced)" if imbalanced else "")
                    + "; chosen before any model runs", [ev], 1.0, "rule",
                    options=sorted({m["primary"] for m in rt.METRICS.values()}), chosen=plan.metric)
        self.finished(f"{family}, target {plan.target}, metric {plan.metric}")
        return plan

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def parse_answer(question: ClarifyingQuestion | dict, choice: str, columns: list[str]) -> dict:
        """Turn the option the client picked (or typed) into a structured answer."""
        reason = question["reason"] if isinstance(question, dict) else question.reason
        c = choice.strip()
        if reason == "rl_mismatch":
            return {"rl_mismatch": "accept_supervised" if c.lower().startswith("yes") else "decline"}
        low = c.lower()
        if "clustering" in low or "segment" in low:
            return {"task_family": "clustering"}
        if "anomaly" in low or "unusual" in low:
            return {"task_family": "anomaly_detection"}
        name = c.split("'")[1] if c.count("'") >= 2 else c
        match = next((col for col in columns if col.lower() == name.lower()), None)
        return {"target": match} if match else {"free_text": c}

    def _question(self, pre: dict, profile: dict, ambiguous) -> ClarifyingQuestion:
        if ambiguous:
            return ClarifyingQuestion(question="Which of these columns do you want to predict?",
                                      options=list(pre["ambiguous_targets"])[:5],
                                      reason=f"the statement names several possible targets: {', '.join(pre['ambiguous_targets'])}",
                                      kind="target")
        opts = [f"Predict '{c}'" for c in pre["hint_targets"][:4]]
        opts += ["Find segments (clustering)", "Find unusual rows (anomaly detection)"]
        why = "; ".join(pre["reasons"]) or "the statement does not name a target or a task"
        return ClarifyingQuestion(question="What should I do with this data: predict a column, or find structure?",
                                  options=opts, reason=why, kind="target")

    def _group_column(self, profile: dict, target: str | None) -> str | None:
        rows = profile["rows"]
        for c in profile["column_info"]:
            n = c["name"].lower()
            if c["name"] != target and (n.endswith("_id") or n.endswith("id") and len(n) > 2) and \
                    1 < c["n_unique"] < 0.5 * rows and c["kind"] in ("id_like", "numeric", "categorical"):
                return c["name"]
        return None

    def _llm_map(self, statement: str, profile: dict, pre: dict) -> str | None:
        cols = [c["name"] for c in profile["column_info"]]
        prompt = ("Map the user's problem statement to ONE column of the dataset that should be predicted, "
                  "or null if no column fits. Reply as JSON {\"target\": <column name or null>}.\n"
                  f"Statement: {statement}\nColumns: {json.dumps(cols)}")
        out = ask_json(self.ctx.llm, prompt)
        t = out.get("target") if isinstance(out, dict) else None
        return t if t in cols else None
