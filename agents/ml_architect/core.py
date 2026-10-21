"""ML Architect: generates candidate pipelines conditioned on upstream findings, evaluates them
with real cross-validation, ranks them, and revises them when the Critic objects.
It never auto-selects: every candidate goes to the Critic and the debate."""
from __future__ import annotations

import copy

import pandas as pd

from agents.base import BaseAgent, cv_claim
from app.schemas import (CandidatePipeline, Check, Claim, CVScores, DebatePosition, Objection,
                         StatisticalProfile, TaskPlan)
from tools import pipeline_core as pc
from tools import sklearn_tools as sk


class MLArchitectAgent(BaseAgent):
    name = "ml_architect"

    def __init__(self, ctx, folds: int = 5, repeats: int = 3):
        super().__init__(ctx)
        self.folds, self.repeats = folds, repeats
        self.baseline: dict | None = None

    # ------------------------------------------------------------------ propose
    def propose(self, df_train: pd.DataFrame, plan: TaskPlan, profile: StatisticalProfile | None,
                memory_suggestions: list[dict] | None = None) -> list[CandidatePipeline]:
        self.started("building candidate pipelines")
        task = plan.model_dump()
        signals = profile.signals if profile else {}
        cluster_choice = None
        if plan.task_family == "clustering":
            cluster_choice, _ = self.ctx.tool("sklearn_tools.choose_clusters", sk.choose_clusters, {"rows": len(df_train)},
                                              df_train, task, self.ctx.seed)
        n_feat = len(pc.feature_columns(df_train, task)["numeric"]) + len(pc.feature_columns(df_train, task)["categorical"])
        fac, ev_fac = self.ctx.tool("sklearn_tools.candidate_specs", sk.candidate_specs,
                                    {"task": plan.task_family, "signals": signals},
                                    task, signals, len(df_train), n_feat, self.ctx.fast, cluster_choice)
        specs = list(fac["specs"])
        for i, m in enumerate(memory_suggestions or []):
            spec = copy.deepcopy(m["spec"])
            spec.update(id=f"m{i + 1}_{spec['model']}", name=f"{spec['model'].replace('_', ' ')} (from memory)",
                        reason=f"a similar dataset (distance {m['distance']:.2f}) used it: test {m['metric']} "
                               f"{m['test_score']:.3f} against baseline {m['baseline_score']:.3f}",
                        interpretable=spec["model"] in pc.INTERPRETABLE, origin="memory")
            specs.append(spec)
        self.baseline, ev_b = self.ctx.tool("sklearn_tools.evaluate_baseline", sk.evaluate_baseline,
                                            {"spec": pc.baseline_spec(task)}, df_train, task, self.ctx.seed,
                                            self.folds, self.repeats)
        cands = [self.evaluate(spec, df_train, plan, origin=spec.get("origin", "factory")) for spec in specs]
        cands = self.rank(cands, plan.metric)
        best = cands[0]
        self.record("candidate_generation",
                    f"{len(cands)} candidates; all go to the Critic (no automatic selection)",
                    f"conditioning: {'; '.join(fac['conditioning']) or 'none'}. Best CV {plan.metric}: "
                    f"{best.name} {best.cv.primary_mean:.4f} vs baseline {self.baseline['primary_mean']:.4f}",
                    [ev_fac, ev_b], 1.0, "rule", options=[c.name for c in cands], chosen="all candidates to critic")
        self.finished(f"{len(cands)} candidates evaluated")
        return cands

    def evaluate(self, spec: dict, df_train: pd.DataFrame, plan: TaskPlan, origin: str = "factory",
                 parent: str | None = None) -> CandidatePipeline:
        task = plan.model_dump()
        res, ev = self.ctx.tool("sklearn_tools.evaluate_pipeline", sk.evaluate_pipeline,
                                {"candidate": spec["id"], "model": spec["model"], "params": spec.get("params", {}),
                                 "folds": self.folds, "repeats": self.repeats},
                                spec, df_train, task, self.ctx.seed, self.folds, self.repeats,
                                self.baseline["primary_mean"] if self.baseline else None)
        cols = pc.feature_columns(df_train, task)
        cv = CVScores(primary=res["primary"], primary_mean=_num(res["primary_mean"]), primary_std=_num(res["primary_std"]),
                      per_fold=[_num(v) for v in res["per_fold"]], secondary={k: _num(v) for k, v in res["secondary"].items()},
                      train_mean=res.get("train_mean"), fit_time=res["fit_time"], n_splits=res["n_splits"],
                      baseline_mean=self.baseline["primary_mean"] if self.baseline else None)
        self.ctx.emit("candidate_evaluated", {"id": spec["id"], "model": spec["model"], "mean": cv.primary_mean,
                                              "std": cv.primary_std, "metric": cv.primary})
        return CandidatePipeline(id=spec["id"], name=spec.get("name", spec["model"]), model=spec["model"],
                                 spec={k: spec.get(k) for k in ("model", "params", "prep", "balance")},
                                 steps=pc.describe_steps(spec, cols), cv=cv, origin=origin, parent=parent,
                                 interpretable=bool(spec.get("interpretable", spec["model"] in pc.INTERPRETABLE)),
                                 rationale=spec.get("reason", ""), eval_run_id=ev.run_id)

    @staticmethod
    def rank(cands: list[CandidatePipeline], metric: str) -> list[CandidatePipeline]:
        return sorted(cands, key=lambda c: -pc.oriented(metric, c.cv.primary_mean) if c.cv.primary_mean == c.cv.primary_mean else 1e9)

    # ------------------------------------------------------------------ revise
    def revise(self, cand: CandidatePipeline, objections: list[Objection], df_train: pd.DataFrame,
               plan: TaskPlan, round_no: int) -> CandidatePipeline | None:
        """Answer the Critic's objections with a concrete change, or return None if nothing helps."""
        spec = copy.deepcopy(cand.spec)
        p, prep = dict(spec.get("params") or {}), dict(spec.get("prep") or {})
        changes = []
        checks = {o.check for o in objections if not o.resolved}
        if "unnecessary_step_check" in checks:
            for k in ("scale", "power", "winsorize"):
                if prep.get(k):
                    prep[k] = False
                    changes.append(f"removed {k}")
        n_rows = len(df_train)
        leaf_cap = max(5, n_rows // 20)
        if checks & {"overfit_check", "shift_sensitivity", "seed_stability"}:
            m = spec["model"]
            if m == "random_forest":
                p["max_depth"] = max(3, min(p.get("max_depth") or 12, 8) - 2 * (round_no - 1))
                p["min_samples_leaf"] = min(leaf_cap, max(p.get("min_samples_leaf", 1), 5) * round_no)
                changes.append(f"limited depth to {p['max_depth']} and leaves to {p['min_samples_leaf']} rows")
            elif m == "xgboost":
                p["max_depth"] = max(2, (p.get("max_depth") or 4) - 1)
                p["min_child_weight"] = (p.get("min_child_weight") or 1) * 3
                p["reg_lambda"] = (p.get("reg_lambda") or 1.0) * 3
                changes.append("shallower trees and stronger L2")
            elif m == "lightgbm":
                p["num_leaves"] = max(7, (p.get("num_leaves") or 31) // 2)
                p["min_child_samples"] = min(leaf_cap, (p.get("min_child_samples") or 20) * 2)
                changes.append("fewer leaves and larger leaf size")
            elif m == "gradient_boosting":
                p["max_depth"] = max(2, (p.get("max_depth") or 6) - 2)
                p["l2_regularization"] = (p.get("l2_regularization") or 0.0) + 1.0
                p["min_samples_leaf"] = min(leaf_cap, (p.get("min_samples_leaf") or 20) * 2)
                changes.append("shallower trees and L2 regularization")
            elif m == "decision_tree":
                p["max_depth"] = max(2, (p.get("max_depth") or 4) - 1)
                changes.append("shallower tree")
            elif m in ("logreg", "logreg_l1"):
                p["C"] = max(0.01, (p.get("C") or 1.0) * 0.3)
                changes.append(f"stronger regularization (C = {p['C']:.3g})")
            elif m in ("ridge", "lasso", "greedy_ridge"):
                p["alpha"] = min(100.0, (p.get("alpha") or 1.0) * 3)
                changes.append(f"stronger regularization (alpha = {p['alpha']:.3g})")
            elif m in ("kmeans",):
                changes.append("no structural change helps k-means stability")
            if "seed_stability" in checks and m in ("random_forest", "xgboost", "lightgbm"):
                key = "n_estimators"
                p[key] = int((p.get(key) or 200) * 2)
                changes.append(f"{p[key]} trees to reduce seed variance")
        real = [c for c in changes if not c.startswith("no structural")]
        if not real:
            return None
        spec.update(params=p, prep=prep, id=f"{cand.id.split('_r')[0]}_r{round_no}",
                    name=f"{cand.name.split(' (rev')[0]} (rev {round_no})", reason="; ".join(real),
                    interpretable=cand.interpretable)
        new = self.evaluate(spec, df_train, plan, origin="revision", parent=cand.id)
        new.rationale = f"revision of {cand.id}: " + "; ".join(real)
        return new

    def position(self, survivors, plan, round_no: int = 0):
        """Prefer the best cross-validated score; confidence = bootstrap P(best beats runner-up)."""
        ranked = MLArchitectAgent.rank(survivors, plan.metric)
        best = ranked[0]
        reasons = [c for c in (cv_claim(self.ctx.log, best, f"{best.id} has the best cross-validated {plan.metric}: "
                                                             f"{best.cv.primary_mean:.4f}"),) if c]
        if len(ranked) > 1:
            ru = ranked[1]
            bb, ev = self.ctx.tool("sklearn_tools.bootstrap_better", sk.bootstrap_better, {"a": best.id, "b": ru.id},
                                   best.cv.per_fold, ru.cv.per_fold, plan.metric, self.ctx.seed)
            reasons.append(Claim(kind="cv", text=f"{best.id} beats runner-up {ru.id} in {bb['p_better']:.0%} of bootstrap resamples "
                                      f"of the paired fold differences", evidence=[ev],
                                 checks=[Check(run_id=ev.run_id, path="p_better", op="approx", value=bb["p_better"])]))
            conf = bb["p_better"]
        else:
            conf = 0.5
        return DebatePosition(agent=self.name, preferred=best.id, reasons=reasons, confidence=round(conf, 4),
                              confidence_method="bootstrap", round=round_no)


def _num(v) -> float:
    return float("nan") if v is None else float(v)
