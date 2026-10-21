"""The main workflow: router -> auditor -> statistician -> architect -> critic -> debate ->
supervisor -> tuning -> final test -> explainability -> report.

Each stage saves its output to the session before the next starts, so a crashed or restarted
run resumes from the first unfinished stage. A pause (approval, clarification, blocker) ends the
run thread cleanly; answering it records the answer and starts a new run that resumes here."""
from __future__ import annotations

import hashlib
import time
import traceback
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from agents.base import AgentContext
from agents.data_auditor.core import DataAuditorAgent
from agents.ml_architect.core import MLArchitectAgent
from agents.pipeline_critic.core import PipelineCriticAgent
from agents.router.core import RouterAgent
from agents.statistician.core import StatisticianAgent
from agents.supervisor.core import SupervisorAgent
from app import events
from app.config import get_settings, package_versions
from app.ledger import Ledger
from app.llm import BaseLLM, get_llm
from app.schemas import ClarifyingQuestion, DataAuditReport, Recommendation, StatisticalProfile, TaskPlan
from app.session import Session, store
from tools import pipeline_core as pc
from tools import sklearn_tools as sk
from tools.common import DataError, jsonable
from tools.pandas_tools import coerce_types, load_csv, quick_profile
from tools.tool_log import ToolLog
from workflows.modelling import critic_loop

STAGES = ["profile", "route", "audit", "blockers", "approve_preprocessing", "clean", "stats", "approve_features",
          "candidates", "critic", "debate", "approve_modelling", "tune", "final", "explain", "report"]
REVIEW_BUNDLES = {"approve_preprocessing": "preprocessing", "approve_features": "feature_engineering",
                  "approve_modelling": "modelling", "tune": "tuning"}


class Pause(Exception):
    def __init__(self, kind: str, key: str, payload: dict):
        super().__init__(f"{kind}:{key}")
        self.kind, self.key, self.payload = kind, key, payload


class TestSetGuard:
    """The held-out test set may be read exactly once, after the winner is chosen."""

    def __init__(self, session: Session):
        self.s = session

    def read(self, purpose: str) -> None:
        n = int(self.s.stage_outputs.get("test_reads", 0))
        if n >= 1:
            raise RuntimeError(f"held-out test set already used once; refusing a second read for '{purpose}'")
        self.s.stage_outputs["test_reads"] = n + 1


class PipelineRun:
    def __init__(self, session_id: str, llm: BaseLLM | None = None, interactive: bool = True,
                 auto_answers: dict[str, Any] | None = None, on_event: Callable[[str, dict], None] | None = None):
        s = store.get(session_id)
        if s is None:
            raise KeyError(session_id)
        self.s = s
        self.settings = get_settings()
        self.llm = llm or get_llm()
        self.interactive = interactive
        self.auto_answers = auto_answers or {}
        self.on_event = on_event
        self.folds = self.settings.cv_folds
        self.repeats = 1 if self.settings.fast else self.settings.cv_repeats
        self.ctx = AgentContext(session_id=s.id, log=ToolLog(s.id, on_run=self._on_tool), ledger=Ledger(s.id, on_add=self._on_ledger),
                                llm=self.llm, emit=self.emit, seed=s.seed, fast=self.settings.fast)
        self._df_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ plumbing
    def emit(self, event_type: str, data: dict | None = None) -> None:
        events.emit(self.s.id, event_type, data or {})
        if self.on_event:
            self.on_event(event_type, data or {})

    def _on_tool(self, tool: str, run_id: str, seconds: float) -> None:
        self.emit("tool_called", {"tool": tool, "run_id": run_id, "seconds": round(seconds, 3), "stage": self.s.current_stage})

    def _on_ledger(self, entry) -> None:
        r = entry.recommendation
        self.emit("ledger_entry", {"step": entry.step, "decision_point": entry.decision_point, "chosen": entry.chosen,
                                   "agent": r.agent, "confidence": r.confidence, "method": r.confidence_method,
                                   "decided_by": entry.decided_by, "overridden": entry.overridden})

    def save(self) -> None:
        store.save(self.s)

    def answer(self, key: str) -> Any:
        return self.s.answers.get(key, self.auto_answers.get(key))

    def ask(self, kind: str, key: str, payload: dict, default: Any) -> Any:
        """Return the client's answer, or pause the run until it arrives (non-interactive runs use the default)."""
        ans = self.answer(key)
        if ans is not None:
            return ans
        if not self.interactive:
            return default
        self.s.pending = {"kind": kind, "key": key, "payload": jsonable(payload)}
        self.s.status = {"approval": "awaiting_approval", "clarification": "awaiting_clarification",
                         "blocker": "blocked"}[kind]
        self.save()
        self.emit({"approval": "approval_required", "clarification": "clarification_required", "blocker": "blocker"}[kind],
                  {"key": key, **jsonable(payload)})
        raise Pause(kind, key, payload)

    def review(self, stage: str) -> bool:
        return self.s.mode == "review" and stage in REVIEW_BUNDLES

    @property
    def plan(self) -> TaskPlan:
        return self.s.task_plan

    def raw_df(self) -> pd.DataFrame:
        if "raw" not in self._df_cache:
            self._df_cache["raw"] = coerce_types(load_csv(self.s.file_path))
        return self._df_cache["raw"]

    def clean_df(self) -> pd.DataFrame:
        if "clean" not in self._df_cache:
            self._df_cache["clean"] = pd.read_pickle(self.s.dir / "clean.pkl")
        return self._df_cache["clean"]

    def split(self) -> tuple[np.ndarray, np.ndarray]:
        z = np.load(self.s.dir / "split.npz")
        return z["train"], z["test"]

    def train_df(self) -> pd.DataFrame:
        if "train" not in self._df_cache:
            tr, _ = self.split()
            self._df_cache["train"] = self.clean_df().iloc[tr].reset_index(drop=True)
        return self._df_cache["train"]

    # ------------------------------------------------------------------ run
    def run(self, stop_after: str | None = None) -> Session:
        s = self.s
        s.status, s.pending, s.error = "running", None, None
        if "versions" not in s.stage_outputs:
            s.stage_outputs["versions"] = package_versions()
        self.save()
        self.emit("status", {"status": "running", "llm": self.llm.name})
        t0 = time.perf_counter()
        try:
            for name in STAGES:
                if name in s.completed_stages:
                    continue
                s.current_stage = name
                self.save()
                self.emit("stage", {"stage": name})
                getattr(self, f"stage_{name}")()
                s.completed_stages.append(name)
                self.save()
                if stop_after == name:
                    break
            else:
                s.status = "completed"
                s.current_stage = None
                self.save()
                self.emit("completed", {"chosen": s.chosen_id, "test": s.results.get("test")})
        except Pause:
            pass
        except DataError as e:
            self._fail(str(e))
        except Exception as e:  # noqa: BLE001 - any failure ends in a readable message and keeps the ledger
            self._fail(f"{type(e).__name__} in stage '{s.current_stage}': {e}", traceback.format_exc(limit=4))
        finally:
            u = s.usage or {}
            for k, v in self.llm.usage.items():
                u[k] = u.get(k, 0) + v
            u["seconds"] = round(u.get("seconds", 0) + time.perf_counter() - t0, 2)
            u["llm"] = self.llm.name
            s.usage = u
            self.save()
        return s

    def _fail(self, message: str, tb: str | None = None) -> None:
        self.s.status, self.s.error = "failed", message
        self.save()
        self.emit("failed", {"message": message, "stage": self.s.current_stage, "trace": tb})

    # ------------------------------------------------------------------ stages
    def stage_profile(self) -> None:
        self.s.profile = quick_profile(self.s.file_path, self.settings.max_profile_rows)

    def stage_route(self) -> None:
        ans = self.s.answers.get("clarify")
        if isinstance(ans, dict) and ans.get("rl_mismatch") == "decline":
            raise DataError("Stopped: the data does not fit offline RL. Upload logged-decision data "
                            "(an action column and a reward column) to run RL.")
        router = RouterAgent(self.ctx)
        out = router.plan(self.s.problem_statement, self.s.profile, self.raw_df(), answer=ans if isinstance(ans, dict) else None)
        if isinstance(out, ClarifyingQuestion):
            auto = self.auto_answers.get("clarify")
            if not self.interactive and auto:   # scripted runs (eval) answer only when the router actually asks
                out = router.plan(self.s.problem_statement, self.s.profile, self.raw_df(), answer=auto)
            if isinstance(out, ClarifyingQuestion):
                if not self.interactive:
                    raise DataError(f"The router needs a clarification: {out.question}")
                self.ask("clarification", "clarify", out.model_dump(), default=None)
        self.s.task_plan = out

    def stage_audit(self) -> None:
        self.s.audit_report = DataAuditorAgent(self.ctx).audit(self.raw_df(), self.plan)

    def stage_blockers(self) -> None:
        rep: DataAuditReport = self.s.audit_report
        if not rep.blockers:
            return
        policy = self.settings.blocker_policy
        default = {c: {"decision": "keep" if policy == "keep" else "drop", "explanation": f"automatic policy: {policy}"}
                   for c in rep.blockers}
        findings = [f.model_dump() for f in rep.findings if f.severity == "blocker"]
        ans = self.ask("blocker", "blockers", {"blockers": rep.blockers, "findings": findings,
                                                "question": "Confirm dropping each column, or explain why it is "
                                                            "available at prediction time."},
                       default if (policy in ("drop", "keep") or not self.interactive) else None)
        for col in rep.blockers:
            a = (ans or {}).get(col, {"decision": "drop"})
            keep = a.get("decision") == "keep"
            for act in rep.cleaning_plan:
                if act.action == "drop_column" and act.column == col:
                    act.enabled = not keep
            f = next(f for f in rep.findings if f.column == col and f.severity == "blocker")
            self.ctx.ledger.add(Recommendation(agent="data_auditor", decision_point=f"blocker:{col}",
                                               recommendation="drop the column", rationale=f.rationale,
                                               evidence=f.evidence, confidence=f.confidence,
                                               confidence_method=f.confidence_method),
                                options=["drop", "keep"], chosen="keep" if keep else "drop", decided_by="client",
                                client_choice=a.get("decision"), overridden=keep, note=a.get("explanation"))

    def stage_approve_preprocessing(self) -> None:
        if not self.review("approve_preprocessing"):
            return
        plan = [a.model_dump() for a in self.s.audit_report.cleaning_plan]
        ans = self.ask("approval", "preprocessing", {"bundle": "preprocessing", "actions": plan,
                                                     "summary": self.s.audit_report.summary}, {"decision": "approve"})
        self._apply_bundle_answer("preprocessing", ans)

    def _apply_bundle_answer(self, bundle: str, ans: dict) -> None:
        disabled = set(ans.get("disabled", [])) if ans.get("decision") == "change" else set()
        if bundle == "preprocessing":
            for i, a in enumerate(self.s.audit_report.cleaning_plan):
                if i in disabled or f"{a.action}:{a.column}" in disabled:
                    a.enabled = False
        ev = self.s.audit_report.findings[0].evidence if self.s.audit_report.findings else []
        if ev:
            self.ctx.ledger.add(Recommendation(agent="supervisor", decision_point=f"approve:{bundle}",
                                               recommendation=f"{bundle} bundle as proposed", rationale="review mode approval",
                                               evidence=ev, confidence=1.0, confidence_method="rule"),
                                options=["approve", "change"], chosen=ans.get("decision", "approve"), decided_by="client",
                                client_choice=ans.get("decision"), overridden=ans.get("decision") == "change",
                                note=f"disabled: {sorted(map(str, disabled))}" if disabled else None)

    def stage_clean(self) -> None:
        plan = [a.model_dump() for a in self.s.audit_report.cleaning_plan]
        res, ev = self.ctx.tool("pipeline_core.apply_cleaning_plan", lambda: _clean_summary(self, plan),
                                {"actions": [f"{a['action']}:{a['column']}" for a in plan if a["enabled"] and a["stage"] == "cleaning"]})
        task = self.plan.model_dump()
        clean = self._df_cache["clean"]
        tr, te = pc.holdout_split(clean, task, self.s.seed)
        np.savez(self.s.dir / "split.npz", train=tr, test=te)
        self.s.stage_outputs["clean"] = {"rows": len(clean), "columns": list(clean.columns), "changes": res["changes"],
                                         "n_train": int(len(tr)), "n_test": int(len(te))}
        self.ctx.ledger.add(Recommendation(agent="data_auditor", decision_point="apply_cleaning_plan",
                                           recommendation=f"{len(res['changes'])} approved actions applied",
                                           rationale="stateless fixes before the split; imputation, capping and encoding "
                                                     "are fitted inside each CV fold", evidence=[ev], confidence=1.0,
                                           confidence_method="rule"),
                            options=["apply", "skip"], chosen="apply")
        self.ctx.ledger.add(Recommendation(agent="supervisor", decision_point="holdout_split",
                                           recommendation=f"hold out {len(te)} rows until the winner is chosen",
                                           rationale="the test set is read once, for the reported score"
                                           + (" (last 20% by time)" if task["task_family"] == "time_series" else ""),
                                           evidence=[ev], confidence=1.0, confidence_method="rule"),
                            options=["holdout 20%"], chosen=f"train {len(tr)} / test {len(te)}")

    def stage_stats(self) -> None:
        train = self.train_df()
        self.s.stats_profile = StatisticianAgent(self.ctx).profile(train, self.plan, self.s.audit_report)
        if self.s.flags.memory:
            from memory.fingerprint_store import lookup_for_run
            lookup_for_run(self)

    def stage_approve_features(self) -> None:
        if not self.review("approve_features"):
            return
        sig = self.s.stats_profile.signals
        keys = ["power_transform", "winsorize", "add_indicator", "class_weight"]
        ans = self.ask("approval", "feature_engineering",
                       {"bundle": "feature_engineering", "options": {k: bool(sig.get(k)) for k in keys},
                        "guidance": self.s.stats_profile.guidance}, {"decision": "approve"})
        if ans.get("decision") == "change":
            for k, v in (ans.get("options") or {}).items():
                if k in keys:
                    sig[k] = bool(v)
        self._apply_bundle_answer("feature_engineering", ans)

    def stage_candidates(self) -> None:
        arch = MLArchitectAgent(self.ctx, self.folds, self.repeats)
        sugg = self.s.memory.get("suggestions") if self.s.flags.memory else None
        self.s.candidates = arch.propose(self.train_df(), self.plan, self.s.stats_profile, sugg)
        self.s.stage_outputs["baseline"] = arch.baseline

    def stage_critic(self) -> None:
        baseline = self.s.stage_outputs["baseline"]["primary_mean"]
        if not self.s.flags.critic:
            return
        arch = MLArchitectAgent(self.ctx, self.folds, self.repeats)
        arch.baseline = self.s.stage_outputs["baseline"]
        top_feature = self._top_feature()
        mem = {c.id: s for c, s in zip([c for c in self.s.candidates if c.origin == "memory"],
                                       self.s.memory.get("suggestions") or [])}
        target_type = self._target_type()
        self.s.candidates = critic_loop(self.ctx, arch, PipelineCriticAgent(self.ctx), self.s.candidates, self.train_df(),
                                        self.plan, baseline, top_feature, top_k=3 if self.settings.fast else 4,
                                        memory_suggestions=mem, target_type=target_type)
        if mem:
            from memory.fingerprint_store import log_memory_review
            log_memory_review(self)

    def _top_feature(self) -> str | None:
        prof: StatisticalProfile = self.s.stats_profile
        eff = [f for f in prof.features if f.effect_size is not None]
        return max(eff, key=lambda f: f.effect_size).name if eff else None

    def _target_type(self) -> str | None:
        from tools.router_tools import target_type
        if not self.plan.target:
            return None
        df = self.train_df()
        return target_type(df, self.plan.target, "numeric")["type"]

    def stage_debate(self) -> None:
        survivors = [c for c in self.s.candidates if c.status == "active"]
        baseline = self.s.stage_outputs["baseline"]["primary_mean"]
        f = self.s.flags
        out = SupervisorAgent(self.ctx).decide(survivors, self.plan, self.s.stats_profile, self.train_df(), baseline,
                                               verify=f.verify, llm_positions=f.llm_positions, debate=f.debate)
        self.s.debate = out
        self.s.chosen_id = out.chosen
        for c in self.s.candidates:
            if c.id in (out.chosen, out.runner_up):
                c.status = "finalist"

    def stage_approve_modelling(self) -> None:
        if not self.review("approve_modelling"):
            return
        rec = self.s.debate.recommendation
        table = [{"id": c.id, "model": c.model, "cv_mean": c.cv.primary_mean, "cv_std": c.cv.primary_std,
                  "objections": [o.message for o in c.objections if not o.resolved], "status": c.status}
                 for c in self.s.candidates if c.status in ("active", "finalist")]
        ans = self.ask("approval", "modelling", {"bundle": "modelling", "recommended": rec.recommendation,
                                                 "confidence": rec.confidence, "rationale": rec.rationale,
                                                 "rule": self.s.debate.rule, "candidates": table}, {"decision": "approve"})
        if ans.get("decision") == "change" and ans.get("choice") and ans["choice"] != self.s.chosen_id:
            self.apply_override(ans["choice"], ans.get("justification"))
        else:
            self._apply_bundle_answer("modelling", {"decision": "approve"})

    def apply_override(self, choice: str, justification: str | None) -> str | None:
        """Client picks another pipeline: warn once (with numbers) if it should underperform, then
        proceed with the client's choice and log both."""
        rec_id = self.s.chosen_id
        if self.s.candidate(choice) is None:
            raise DataError(f"Unknown pipeline '{choice}'")
        key = f"override:{choice}"
        warning = None if key in self.s.warnings_given else override_warning(self.s, choice)
        if warning:
            self.s.warnings_given.append(key)
            self.emit("warning", {"message": warning})
        b = self.s.candidate(choice)
        ev = self.s.debate.recommendation.evidence
        self.ctx.ledger.add(Recommendation(agent="supervisor", decision_point="model_choice_override",
                                           recommendation=rec_id, rationale=warning or "client override within noise",
                                           evidence=ev, confidence=self.s.debate.recommendation.confidence,
                                           confidence_method="bootstrap"),
                            options=[c.id for c in self.s.candidates if c.status in ("active", "finalist")], chosen=choice,
                            decided_by="client", client_choice=choice, overridden=True,
                            note=f"justification: {justification}" if justification else "no justification given")
        self.s.results["override"] = {"recommended": rec_id, "client": choice, "warning": warning,
                                      "rule": self.s.debate.rule, "justification": justification}
        self.s.chosen_id = choice
        if b.status != "finalist":
            b.status = "finalist"
        return warning

    def stage_tune(self) -> None:
        metric, chosen = self.plan.metric, self.s.candidate(self.s.chosen_id)
        runner_id = self.s.debate.runner_up if not self.s.results.get("override") else self.s.results["override"]["recommended"]
        if not self.s.flags.tuning:
            self._final_recommendation(chosen, self.s.candidate(runner_id) if runner_id else None, "no tuning (ablation)")
            return
        arch = MLArchitectAgent(self.ctx, self.folds, self.repeats)
        arch.baseline = self.s.stage_outputs["baseline"]
        tuned = {}
        trials = 8 if self.settings.fast else 20
        for cid in [c for c in (self.s.chosen_id, runner_id) if c]:
            if f"tuned:{cid}" in self.s.stage_outputs:
                tuned[cid] = self.s.candidate(self.s.stage_outputs[f"tuned:{cid}"])
                continue
            c = self.s.candidate(cid)
            spec = {**c.spec, "id": f"{cid.split('_t')[0]}_tuned", "name": f"{c.name} (tuned)",
                    "interpretable": c.interpretable}
            res, ev = self.ctx.tool("sklearn_tools.tune_pipeline", sk.tune_pipeline,
                                    {"candidate": cid, "n_trials": trials}, spec, self.train_df(), self.plan.model_dump(),
                                    self.s.seed, trials, 45.0 if self.settings.fast else 120.0)
            if not res.get("tuned"):
                continue
            spec["params"] = res["best_params"]
            t = arch.evaluate(spec, self.train_df(), self.plan, origin="tuned", parent=cid)
            t.status = "finalist"
            self.s.candidates.append(t)
            self.s.stage_outputs[f"tuned:{cid}"] = t.id
            tuned[cid] = t
        self.save()
        best_of = lambda x, y: x if y is None or pc.oriented(metric, x.cv.primary_mean) >= pc.oriented(metric, y.cv.primary_mean) else y
        final = best_of(chosen, tuned.get(chosen.id))
        note = f"tuned version {'kept' if final is not chosen else 'did not beat the default'}"
        other = None
        if runner_id:
            r = best_of(self.s.candidate(runner_id), tuned.get(runner_id))
            other = r
            if not self.s.results.get("override") and \
                    pc.oriented(metric, r.cv.primary_mean) - pc.oriented(metric, final.cv.primary_mean) > final.cv.primary_std:
                note = (f"loop-back: tuned runner-up {r.id} beats {final.id} by more than one standard deviation "
                        f"({r.cv.primary_mean:.4f} vs {final.cv.primary_mean:.4f}); it becomes the front-runner")
                final, other = r, final
        if self.review("tune"):
            ans = self.ask("approval", "tuning", {"bundle": "tuning", "default": chosen.model_dump(include={"id", "cv"}),
                                                  "tuned": final.model_dump(include={"id", "cv", "spec"}), "note": note},
                           {"decision": "approve"})
            if ans.get("decision") == "change":
                final, note = chosen, "client kept the untuned pipeline"
        self._final_recommendation(final, other, note)

    def _final_recommendation(self, final, other, note: str) -> None:
        baseline = self.s.stage_outputs["baseline"]["primary_mean"]
        conf, ev = self.ctx.tool("sklearn_tools.evidence_confidence", sk.evidence_confidence,
                                 {"chosen": final.id, "runner_up": other.id if other else None},
                                 final.cv.model_dump(), other.cv.model_dump() if other else None, baseline,
                                 self.plan.metric, self.s.seed)
        self.s.chosen_id = final.id
        final.status = "chosen"
        self.s.results["final_recommendation"] = {"chosen": final.id, "runner_up": other.id if other else None,
                                                  "confidence": conf, "note": note}
        self.ctx.ledger.add(Recommendation(agent="supervisor", decision_point="final_pipeline", recommendation=final.id,
                                           rationale=note, evidence=[ev], confidence=round(conf["confidence"], 4),
                                           confidence_method="bootstrap"),
                            options=[c for c in {final.id, other.id if other else final.id}], chosen=final.id,
                            decided_by="supervisor")

    def stage_final(self) -> None:
        TestSetGuard(self.s).read("final evaluation")
        self.save()
        task, seed = self.plan.model_dump(), self.s.seed
        df, split = self.clean_df(), self.split()
        chosen = self.s.candidate(self.s.chosen_id)
        fr = self.s.results["final_recommendation"]
        runner = self.s.candidate(fr["runner_up"]) if fr.get("runner_up") else None
        holder: dict = {}

        def _score_all():
            out = pc.fit_and_score(df, task, chosen.spec, seed, split=split)
            holder.update(out)
            base = pc.fit_and_score(df, task, pc.baseline_spec(task), seed, split=split, keep_model=False)
            res = {"metric": task["metric"], "chosen": chosen.id, "test": out["test"], "score": out["score"],
                   "n_train": out["n_train"], "n_test": out["n_test"], "baseline": {"test": base["test"], "score": base["score"]}}
            if runner:
                rr = pc.fit_and_score(df, task, runner.spec, seed, split=split, keep_model=False)
                res["runner_up"] = {"id": runner.id, "test": rr["test"], "score": rr["score"]}
            ov = self.s.results.get("override")
            if ov:   # shadow run: both pipelines on the held-out split
                rec = self.s.candidate(ov["recommended"])
                rs = pc.fit_and_score(df, task, rec.spec, seed, split=split, keep_model=False)
                res["shadow"] = {"recommended": ov["recommended"], "recommended_score": rs["score"],
                                 "client": chosen.id, "client_score": out["score"]}
            return res

        res, ev = self.ctx.tool("pipeline_core.fit_and_score", _score_all,
                                {"chosen": chosen.id, "spec": chosen.spec, "seed": seed})
        metric = task["metric"]
        beat_base = pc.oriented(metric, res["score"]) > pc.oriented(metric, res["baseline"]["score"])
        noise = runner.cv.primary_std if runner else 0.0
        beat_runner = (not runner) or pc.oriented(metric, res["score"]) >= pc.oriented(metric, res["runner_up"]["score"]) - noise
        self.s.results.update(test=res, predictions=jsonable(holder.get("predictions")),
                              calibration={"confidence": fr["confidence"]["confidence"],
                                           "outcome": bool(beat_base and beat_runner),
                                           "beat_baseline": bool(beat_base), "beat_runner_up_within_noise": bool(beat_runner)})
        if "model" in holder:
            path = self.s.dir / "model.joblib"
            joblib.dump(holder["model"], path)
            self.s.results["model_file"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            self._df_cache["fit"] = holder
        self.ctx.ledger.add(Recommendation(agent="supervisor", decision_point="final_test_score",
                                           recommendation=f"{metric} {res['score']:.4f} on {res['n_test']} held-out rows",
                                           rationale=f"baseline {res['baseline']['score']:.4f}"
                                           + (f"; runner-up {res['runner_up']['score']:.4f}" if runner else ""),
                                           evidence=[ev], confidence=1.0, confidence_method="rule"),
                            options=[chosen.id], chosen=chosen.id, decided_by="system")

    def stage_explain(self) -> None:
        from workflows.report_graph import explain_stage
        explain_stage(self)

    def stage_report(self) -> None:
        from workflows.report_graph import report_stage
        report_stage(self)


def override_warning(s: Session, choice: str) -> str | None:
    """The one warning a client gets when overriding: only if the stored CV scores say their choice
    should underperform the recommendation by more than one standard deviation."""
    a, b = s.candidate(s.chosen_id), s.candidate(choice)
    if a is None or b is None or a.id == b.id:
        return None
    metric = s.task_plan.metric
    gap = pc.oriented(metric, a.cv.primary_mean) - pc.oriented(metric, b.cv.primary_mean)
    if gap <= a.cv.primary_std:
        return None
    return (f"{choice} is expected to underperform: cross-validated {metric} {b.cv.primary_mean:.4f} against "
            f"{a.cv.primary_mean:.4f} for the recommended {a.id} (gap {gap:.4f}, larger than one standard deviation "
            f"{a.cv.primary_std:.4f}). Proceeding with your choice.")


def _clean_summary(run: PipelineRun, plan: list[dict]) -> dict:
    clean, log = pc.apply_cleaning_plan(run.raw_df(), plan)
    clean.to_pickle(run.s.dir / "clean.pkl")
    run._df_cache["clean"] = clean
    return {"rows_before": len(run.raw_df()), "rows_after": len(clean), "changes": log}
