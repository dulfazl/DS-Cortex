"""Explain and report stages: SHAP (with fallback), error analysis, plots, narratives, the report,
the exported code and the export bundle."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from agents.business.core import BusinessInsightAgent
from agents.explainability.core import ExplainabilityAgent
from agents.report_generator.core import ReportGeneratorAgent
from agents.research.core import ResearchAgent
from app.config import get_settings
from app.schemas import Recommendation
from tools import error_tools as et
from tools import export_tools as xt
from tools import pipeline_core as pc
from tools import plotting_tools as plt
from tools import shap_tools as sh
from tools import stats_tools as stt
from tools.common import jsonable, now_iso

if TYPE_CHECKING:
    from workflows.pipeline_graph import PipelineRun


def _fit(run: "PipelineRun") -> dict:
    """The fitted final pipeline and its test predictions (refit from the saved model if resuming)."""
    if "fit" in run._df_cache:
        return run._df_cache["fit"]
    s = run.s
    out = {"predictions": s.results.get("predictions")}
    mf = s.results.get("model_file")
    if mf:
        out["model"] = joblib.load(mf["path"])
        task = s.task_plan.model_dump()
        if task["task_family"] in ("classification", "regression"):
            tr, te = run.split()
            df = run.clean_df()
            train = df.iloc[tr].reset_index(drop=True)
            test = df.iloc[te].reset_index(drop=True)
            info = pc.prepare_supervised(train, task)
            out["X_train"], out["y_train"] = info["X"], info["y"]
            out["X_test"] = test[test[task["target"]].notna()][list(info["X"].columns)].reset_index(drop=True)
        elif task["task_family"] == "clustering":
            out["X_matrix"] = None
    run._df_cache["fit"] = out
    return out


def explain_stage(run: "PipelineRun") -> None:
    s, ctx, task = run.s, run.ctx, run.s.task_plan.model_dump()
    fam = task["task_family"]
    fit = _fit(run)
    explain: dict = {}
    seed = s.seed
    force_fail = s.flags.force_shap_failure
    if fam in ("classification", "regression") and fit.get("model") is not None and s.flags.explain:
        imp, ev = ctx.tool("shap_tools.shap_summary", lambda: sh.shap_summary(fit["model"], fit["X_train"], fit["y_train"],
                                                                               task, seed, force_fail=force_fail),
                           {"model": s.chosen_id, "rows": int(min(len(fit["X_train"]), 400))})
        explain["importance"] = imp
        if imp.get("substituted"):
            ctx.ledger.add(Recommendation(agent="explainability", decision_point="explainer_substitution",
                                          recommendation="permutation importance instead of SHAP",
                                          rationale=imp["substitution_reason"], evidence=[ev], confidence=1.0,
                                          confidence_method="rule"),
                           options=["shap", "permutation_importance"], chosen="permutation_importance", decided_by="system")
        lc, _ = ctx.tool("error_tools.learning_curve", et.learning_curve, {"model": s.chosen_id},
                         s.candidate(s.chosen_id).spec, run.train_df(), task, seed)
        explain["learning_curve"] = lc
    err, _ = ctx.tool("error_tools.error_analysis", et.error_analysis, {"family": fam}, task, s.results["predictions"])
    explain["errors"] = err
    # ---- plots
    train = run.train_df()
    num = stt.numeric_features(train, {task.get("target")})
    corr = stt.correlation_report(train, num) if len(num) >= 2 else None
    top = [f.name for f in sorted([f for f in s.stats_profile.features if f.effect_size is not None],
                                  key=lambda f: -f.effect_size)] or num[:4]
    plots = plt.overview_plots(run.clean_df(), task.get("target") if fam != "clustering" else None, corr, top)
    if fam == "classification":
        plots += plt.classification_plots(err)
    elif fam == "regression":
        plots += plt.regression_plots(err)
    elif fam == "clustering":
        prof, _ = ctx.tool("error_tools.cluster_profiles", et.cluster_profiles, {}, run.clean_df(),
                           s.results["predictions"]["labels"], task, seed)
        explain["cluster_profiles"] = {k: v for k, v in prof.items() if k in ("profiles", "features")}
        plots += plt.clustering_plots(prof)
    elif fam == "anomaly_detection":
        plots += plt.anomaly_plots(err)
    elif fam == "time_series":
        plots += plt.time_series_plots(err)
    if explain.get("importance"):
        plots += plt.explain_plots(explain["importance"], explain.get("learning_curve"))
    plots = plt.rank_plots(plots, s.problem_statement)
    for p in plots:
        run.emit("plot_ready", {"id": p["id"], "title": p["title"], "relevance": p["relevance"]})
    explain["plots"] = plots
    # ---- narratives
    entries = ctx.ledger.entries()
    exp_agent = ExplainabilityAgent(ctx)
    explain["narrative"] = exp_agent.narrate_decisions(entries, "technical")
    explain["narrative_beginner"] = exp_agent.narrate_decisions(entries, "beginner")
    steps = {e.decision_point: e.step for e in entries}
    explain["business"] = BusinessInsightAgent(ctx).stakeholder_summary(s.task_plan, s.results["test"], explain.get("importance"),
                                                                          s.stats_profile, s.audit_report, steps)
    explain["research"] = ResearchAgent(ctx).context_note(s.task_plan, s.stats_profile)
    (s.dir / "explain.json").write_text(json.dumps(jsonable(explain)))
    s.results["explain_summary"] = {"importance_method": (explain.get("importance") or {}).get("method"),
                                    "shap_substituted": (explain.get("importance") or {}).get("substituted", False),
                                    "plots": [p["id"] for p in plots]}


def load_explain(s) -> dict:
    p = s.dir / "explain.json"
    return json.loads(p.read_text()) if p.exists() else {}


def report_stage(run: "PipelineRun") -> None:
    s, ctx = run.s, run.ctx
    explain = load_explain(s)
    gen = ReportGeneratorAgent(ctx)
    gen.started("assembling the report and the export bundle")
    entries = ctx.ledger.entries()
    sec = gen.sections(s, explain, entries)
    out = s.dir / "export"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "model").mkdir(exist_ok=True)
    md = gen.markdown(sec)
    (out / "report.md").write_text(md)
    (out / "report.html").write_text(gen.html(sec, explain.get("plots", [])))
    for p in explain.get("plots", []):
        (out / "plots" / f"{p['id']}.json").write_text(json.dumps(p["figure"]))
    png_note = _static_plots(explain.get("plots", []), out / "plots")
    pdf_ok = _pdf(gen.html(sec, [], static=True), out / "report.pdf")
    # ---- code, model, ledger, transcript, card, manifest
    chosen = s.candidate(s.chosen_id)
    cleaning = [a.model_dump() for a in s.audit_report.cleaning_plan]
    code = xt.export_pipeline_code(s.task_plan.model_dump(), cleaning, chosen.spec, s.seed, s.file_name, s.results["test"]["score"])
    (out / "pipeline.py").write_text(code)
    versions = s.stage_outputs.get("versions", {})
    (out / "requirements.txt").write_text(xt.requirements_for(versions, chosen.model))
    (out / "ledger.json").write_text(ctx.ledger.export("json"))
    (out / "ledger.md").write_text(ctx.ledger.export("markdown"))
    (out / "debate_transcript.md").write_text("# Debate transcript\n\n" + "\n".join(
        f"- round {t['round']} · {t['agent']} · {t['kind']}: {t['text']}" for t in sec["debate"]["transcript"]) + "\n")
    model_note = _save_model(run, out / "model")
    (out / "model" / "model_card.md").write_text(xt.model_card(s, {}, explain.get("importance")))
    schema = _input_schema(run)
    (out / "model" / "input_schema.json").write_text(json.dumps(schema, indent=2))
    settings = get_settings()
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {"session_id": s.id, "created_at": now_iso(), "dataset_file": s.file_name,
                "dataset_sha256": xt.sha256(Path(s.file_path)), "seed": s.seed,
                "versions": versions, "llm": s.usage.get("llm") or run.llm.name,
                "gemini_model": settings.gemini_model if run.llm.available else None, "ds_cortex_commit": xt.git_commit(),
                "run_mode": s.mode, "flags": s.flags.model_dump(),
                "usage": {k: s.usage.get(k, 0) + run.llm.usage.get(k, 0) for k in ("calls", "prompt_tokens", "output_tokens")},
                "tool_runs": len(ctx.log.all()), "notes": [n for n in (png_note, model_note) if n],
                "pdf": "included" if pdf_ok else "not generated (PDF engine unavailable)",
                "files": {str(p.relative_to(out)): xt.sha256(p) for p in files}}
    (out / "manifest.json").write_text(json.dumps(jsonable(manifest), indent=2))
    zip_path = xt.write_zip({str(p.relative_to(out)): p for p in sorted(out.rglob("*")) if p.is_file()},
                            s.dir / f"ds_cortex_{s.id[:8]}.zip")
    s.results["export"] = {"dir": str(out), "zip": str(zip_path), "pdf": pdf_ok, "files": list(manifest["files"]) + ["manifest.json"]}
    s.results["report_markdown"] = md
    from memory.fingerprint_store import record_outcome
    s.memory["outcome"] = record_outcome(run)
    gen.finished("report and bundle ready")


def _static_plots(plots: list[dict], folder) -> str | None:
    try:
        import plotly.io as pio
        for p in plots[:12]:
            pio.write_image(p["figure"], folder / f"{p['id']}.png", width=900, height=420, scale=1)
        return None
    except Exception as e:  # noqa: BLE001 - kaleido needs Chrome; the JSON plots are always included
        return f"static PNG plots skipped ({type(e).__name__}); Plotly JSON is included"


def _pdf(html_text: str, path) -> bool:
    try:
        from xhtml2pdf import pisa
        with open(path, "wb") as f:
            res = pisa.CreatePDF(html_text, dest=f)
        return not res.err
    except Exception:  # noqa: BLE001
        return False


def _save_model(run: "PipelineRun", folder) -> str | None:
    fit = _fit(run)
    model = fit.get("model")
    if model is None:
        return "no fitted model object for this task family (the script refits it)"
    try:
        import skops.io as sio
        sio.dump(model, folder / "model.skops")
        return None
    except Exception as e:  # noqa: BLE001
        joblib.dump(model, folder / "model.joblib")
        return (f"saved with joblib ({type(e).__name__} from skops); load only files you produced yourself: "
                "unpickling can run code")


def _input_schema(run: "PipelineRun") -> dict:
    df, task = run.clean_df(), run.s.task_plan.model_dump()
    cols = pc.feature_columns(df, task)
    out = {"target": task.get("target"), "columns": {}}
    for c in cols["numeric"] + cols["categorical"] + cols["high_cardinality"] + cols["datetime"]:
        rec = {"dtype": str(df[c].dtype)}
        if c in cols["categorical"]:
            rec["levels"] = sorted(map(str, df[c].dropna().unique()))[:100]
        out["columns"][c] = rec
    return out
