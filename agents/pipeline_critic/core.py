"""Pipeline Critic: attacks each candidate before any is chosen. Every objection is a Claim
backed by a tool result and a machine-checkable condition, so the verifier can confirm it."""
from __future__ import annotations

import pandas as pd

from agents.base import BaseAgent
from app.schemas import CandidatePipeline, Check, Claim, Evidence, Objection, TaskPlan
from tools import critic_tools as ct


class PipelineCriticAgent(BaseAgent):
    name = "pipeline_critic"

    def review(self, cand: CandidatePipeline, df_train: pd.DataFrame, plan: TaskPlan, baseline_mean: float,
               top_feature: str | None = None, only: set[str] | None = None, seed_offset: int = 0) -> list[Objection]:
        """Run the checks (all, or `only` the ones that objected last round) and return objections.
        A rebuttal re-runs a check with `seed_offset` to get new evidence."""
        task, spec, seed, tool = plan.model_dump(), cand.spec, self.ctx.seed + seed_offset, self.ctx.tool
        cv = cand.cv.model_dump()
        objs: list[Objection] = []
        run = (lambda name: only is None or name in only)
        fam = plan.task_family
        ident = {"candidate": cand.id, "model": cand.model}

        def obj(check: str, sev: str, msg: str, ev: Evidence, path: str, op: str, value=None):
            claim = Claim(kind="validity", text=f"{cand.id}: {msg}", evidence=[ev],
                          checks=[Check(run_id=ev.run_id, path=path, op=op, value=value)])
            objs.append(Objection(check=check, severity=sev, message=msg, claim=claim))

        if run("unnecessary_step_check"):
            r, ev = tool("critic_tools.unnecessary_step_check", ct.unnecessary_step_check, ident, spec)
            if r["flag"]:
                obj("unnecessary_step_check", "low", "unnecessary steps: " + ", ".join(r["unnecessary"]), ev, "flag", "true")
        if fam in ("classification", "regression"):
            if run("overfit_check"):
                r, ev = tool("critic_tools.overfit_check", ct.overfit_check, {**ident, "seed": seed}, spec, df_train, task, seed,
                             self.ctx.fast)
                if r.get("overfit"):
                    obj("overfit_check", r["severity"],
                        f"train {r['train_mean']:.3f} vs validation {r['val_mean']:.3f}: gap {r['gap']:.3f} above "
                        f"{r['gap_threshold']}", ev, "gap", "gt", r["gap_threshold"])
            if run("leakage_probe"):
                r, ev = tool("critic_tools.leakage_probe", ct.leakage_probe, {**ident, "top_feature": top_feature},
                             spec, df_train, task, seed, baseline_mean, cv["primary_mean"], top_feature)
                if r.get("shuffled_above_baseline"):
                    obj("leakage_probe", "blocker",
                        f"with a shuffled target the score stays at {r['shuffled_mean']:.3f} (chance {r['chance_reference']:.3f}), "
                        f"keeping {r['shuffled_retained_gain']:.0%} of the gain: the pipeline learns something that is "
                        "not the target", ev, "shuffled_above_baseline", "true")
                elif r.get("near_perfect") and not r.get("collapse"):
                    obj("leakage_probe", "high",
                        f"near-perfect cross-validated score ({cv['primary_mean']:.3f}): verify no feature leaks the target",
                        ev, "near_perfect", "true")
                if r.get("collapse"):
                    obj("leakage_probe", "high",
                        f"dropping '{r['top_feature']}' removes {r['share_of_gain_from_top']:.0%} of the gain over baseline: "
                        "one feature carries the model; check it is available at prediction time", ev, "collapse", "true")
            if run("subgroup_check"):
                r, ev = tool("critic_tools.subgroup_check", ct.subgroup_check, ident, spec, df_train, task, seed)
                if r.get("flag"):
                    obj("subgroup_check", "medium",
                        f"segment '{r['worst_segment']}' of '{r['segment_column']}' scores {r['worst_drop']:.3f} below "
                        "the overall score", ev, "worst_drop", "gt", r["threshold"])
            if run("shift_sensitivity") and not self.ctx.fast:
                r, ev = tool("critic_tools.shift_sensitivity", ct.shift_sensitivity, ident, spec, df_train, task, seed,
                             cv["primary_mean"])
                if r.get("flag"):
                    obj("shift_sensitivity", "low",
                        f"on the top 20% of '{r['feature']}' the score drops by {r['drop']:.3f}: fragile under covariate shift",
                        ev, "drop", "gt", r["threshold"])
        if run("seed_stability") and fam in ("classification", "regression", "clustering", "anomaly_detection"):
            r, ev = tool("critic_tools.seed_stability", ct.seed_stability, ident, spec, df_train, task, seed, 3,
                         cv["primary_std"])
            if r["unstable"]:
                obj("seed_stability", "medium",
                    f"mean score varies by {r['relative_sd']:.3f} across seeds {r['seeds']} (limit {r['threshold']:.3f})",
                    ev, "unstable", "true")
        if run("baseline_check"):
            r, ev = tool("critic_tools.baseline_check", ct.baseline_check, ident, cv, baseline_mean, plan.metric)
            if not r["beats_baseline"]:
                obj("baseline_check", "high",
                    f"does not beat the baseline by more than one CV standard deviation (gain {r['gain']:.4f}, "
                    f"sd {r['cv_std']:.4f})", ev, "beats_baseline", "false")
        if run("overlap_check") and fam == "offline_rl":
            r, ev = tool("critic_tools.overlap_check", ct.overlap_check, ident, df_train, task)
            if r.get("flag"):
                obj("overlap_check", "medium",
                    f"{r['share_below_0.05']:.1%} of logged propensities are below 0.05: IPW estimates have high variance",
                    ev, "flag", "true")
        self.ctx.emit("critic_review", {"candidate": cand.id, "objections": [o.check for o in objs]})
        return objs

    def review_memory_suggestion(self, cand: CandidatePipeline, suggestion: dict, plan: TaskPlan,
                                 target_type: str | None, baseline_mean: float) -> Objection | None:
        """A suggestion from memory is advice: reject it if the analogy does not hold."""
        r, ev = self.ctx.tool("critic_tools.memory_fit_check", _memory_fit, {"candidate": cand.id},
                              suggestion, plan.task_family, target_type, cand.cv.model_dump(), baseline_mean, plan.metric)
        if r["reject"]:
            claim = Claim(kind="validity", text=f"{cand.id}: {r['reason']}", evidence=[ev],
                          checks=[Check(run_id=ev.run_id, path="reject", op="true")])
            return Objection(check="memory_fit_check", severity="high", message=r["reason"], claim=claim)
        return None


def _memory_fit(suggestion: dict, task_family: str, target_type: str | None, cv: dict, baseline_mean: float, metric: str) -> dict:
    from tools import pipeline_core as pc
    reasons = []
    if suggestion.get("task_family") != task_family:
        reasons.append(f"the remembered dataset was {suggestion.get('task_family')}, this one is {task_family}")
    if target_type and suggestion.get("target_type") and suggestion["target_type"] != target_type:
        reasons.append(f"the remembered target was {suggestion['target_type']}, this one is {target_type}")
    gain = pc.oriented(metric, cv["primary_mean"]) - pc.oriented(metric, baseline_mean)
    if gain <= 0:
        reasons.append("on this data it does not beat the baseline")
    return {"reject": bool(reasons), "reason": "; ".join(reasons) or "analogy holds", "gain": gain,
            "distance": suggestion.get("distance")}
