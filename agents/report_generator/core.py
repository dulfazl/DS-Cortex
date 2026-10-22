"""Report generator: Markdown and HTML (PDF from the HTML when possible). The report contains the
problem and task plan, the audit summary, what was done and why (from the ledger), the debate
transcript, the client's choices against the recommendations, plots, test results with the
baseline, limitations and the downloadable code. All model-written text is escaped in HTML."""
from __future__ import annotations

import html
import json

from agents.base import BaseAgent

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
LIMITS = [
    "The held-out test set is a single random (or time-ordered) split; scores on new data will vary.",
    "Confidence values come from a hand-designed formula (stability, bootstrap separation, effect over baseline) and are "
    "checked for calibration in the evaluation suite, not guaranteed.",
    "The drift check is a two-sample test against the training data; it does not retrain the model or monitor live data.",
    "Memory suggestions are advisory and come from a small store of past runs.",
    "Research notes are context, not a literature review.",
]


def _e(s) -> str:
    return html.escape("" if s is None else str(s))


class ReportGeneratorAgent(BaseAgent):
    name = "report_generator"

    def sections(self, s, explain: dict, ledger_entries) -> dict:
        plan, test = s.task_plan, s.results["test"]
        fr = s.results["final_recommendation"]
        rec = s.debate.recommendation if s.debate else None
        choices = [{"step": e.step, "decision": e.decision_point, "recommended": e.recommendation.recommendation,
                    "client_choice": e.client_choice, "note": e.note}
                   for e in ledger_entries if e.decided_by == "client"]
        return {
            "title": f"DS Cortex report: {s.file_name}",
            "problem": s.problem_statement,
            "plan": {"task_family": plan.task_family, "target": plan.target, "metric": plan.metric,
                     "secondary": plan.secondary_metrics, "rationale": plan.rationale, "confidence": plan.confidence},
            "audit": {"summary": s.audit_report.summary,
                      "findings": [{"severity": f.severity, "issue": f.issue_type, "column": f.column,
                                    "action": f.recommended_action, "rationale": f.rationale,
                                    "confidence": f.confidence, "method": f.confidence_method} for f in s.audit_report.findings]},
            "guidance": s.stats_profile.guidance if s.stats_profile else [],
            "decisions": explain.get("narrative", {}),
            "narrative_beginner": explain.get("narrative_beginner", {}),
            "candidates": [{"id": c.id, "model": c.model, "origin": c.origin, "status": c.status,
                            "cv_mean": c.cv.primary_mean, "cv_std": c.cv.primary_std,
                            "objections": [f"[{o.severity}] {o.message}" for o in c.objections]}
                           for c in s.candidates],
            "debate": {"rule": s.debate.rule if s.debate else None, "forced_tiebreak": s.debate.forced_tiebreak if s.debate else False,
                       "transcript": [{"round": t.round, "agent": t.agent, "kind": t.kind, "text": t.text}
                                      for t in (s.debate.transcript if s.debate else [])],
                       "verifier": {k: v for k, v in (s.debate.verifier if s.debate else {}).items() if k in
                                    ("total", "verified", "rejected", "rejected_llm", "total_llm")},
                       "notes": s.debate.notes if s.debate else []},
            "recommendation": {"chosen": s.chosen_id, "debate_choice": rec.recommendation if rec else None,
                               "rationale": rec.rationale if rec else "", "confidence": fr["confidence"]["confidence"],
                               "components": {k: fr["confidence"][k] for k in ("stability", "separation", "effect")},
                               "tuning_note": fr["note"]},
            "client_choices": choices,
            "test": test,
            "calibration": s.results.get("calibration"),
            "importance": explain.get("importance"),
            "business": explain.get("business"),
            "research": explain.get("research"),
            "memory": s.memory,
            "plots": [{"id": p["id"], "title": p["title"], "caption": p["caption"]} for p in explain.get("plots", [])],
            "limits": LIMITS + ([f"SHAP was substituted: {explain['importance']['substitution_reason']}"]
                                if explain.get("importance", {}).get("substituted") else []),
        }

    def markdown(self, sec: dict) -> str:
        t = sec["test"]
        L = [f"# {sec['title']}", "", f"**Problem:** {sec['problem']}", "",
             "## Task plan", "", f"- Task family: {sec['plan']['task_family']}; target: {sec['plan']['target'] or 'none'}",
             f"- Primary metric (fixed before modelling): {sec['plan']['metric']}; secondary: {', '.join(sec['plan']['secondary']) or 'none'}",
             f"- Why: {sec['plan']['rationale']} (confidence {sec['plan']['confidence']:.2f})", "",
             "## Result on the held-out test set", "",
             f"| Pipeline | {t['metric']} |", "| --- | --- |", f"| {t['chosen']} (chosen) | {t['score']:.4f} |"]
        if t.get("runner_up"):
            L.append(f"| {t['runner_up']['id']} (runner-up) | {t['runner_up']['score']:.4f} |")
        L.append(f"| baseline | {t['baseline']['score']:.4f} |")
        if t.get("shadow"):
            L.append(f"| shadow: recommended {t['shadow']['recommended']} | {t['shadow']['recommended_score']:.4f} |")
        vs = t.get("vs_baseline") or {}
        if vs.get("applicable"):
            lo, hi = vs["ci95"]
            L += ["", f"Gain over the baseline: {vs['gain']:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}, {vs['method']}): "
                      + ("more than chance." if vs["meaningful"] else "**not more than chance**: the model has no signal to act on.")]
        r = sec["recommendation"]
        L += ["", f"Confidence in the choice: **{r['confidence']:.2f}** (stability {r['components']['stability']:.2f}, "
                  f"separation {r['components']['separation']:.2f}, effect {r['components']['effect']:.2f}). {r['tuning_note']}.", ""]
        if sec.get("business"):
            b = sec["business"]
            L += ["## Summary for stakeholders", "", b["headline"], ""]
            L += [f"- {x['text']} *(source: {x['cites']})*" for x in b["recommendations"]]
            L += [f"- Caveat: {c}" for c in b["caveats"]] + [""]
        L += ["## Data audit", "", sec["audit"]["summary"], "", "| Severity | Issue | Column | Action | Confidence |",
              "| --- | --- | --- | --- | --- |"]
        L += [f"| {f['severity']} | {f['issue']} | {f['column'] or ''} | {f['action']} | {f['confidence']:.2f} ({f['method']}) |"
              for f in sec["audit"]["findings"]]
        L += ["", "## Statistical guidance", ""] + [f"- {g}" for g in sec["guidance"]] + [""]
        L += ["## What was done and why", "", sec["decisions"].get("summary", ""), ""]
        L += [f"{i['step']}. {i['text']}" for i in sec["decisions"].get("items", [])] + [""]
        L += ["## Candidates", "", "| Candidate | Origin | Status | CV mean | CV sd | Objections |", "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| {c['id']} | {c['origin']} | {c['status']} | {c['cv_mean']:.4f} | {c['cv_std']:.4f} | {'; '.join(c['objections']) or '-'} |"
              for c in sec["candidates"]]
        d = sec["debate"]
        L += ["", "## Debate transcript", "", f"Rule applied: **{d['rule']}**" + (" (forced tiebreak)" if d["forced_tiebreak"] else "")
              + f". Verifier: {d['verifier'].get('verified', 0)} of {d['verifier'].get('total', 0)} claims verified.", ""]
        L += [f"- round {x['round']} · {x['agent']} · {x['kind']}: {x['text']}" for x in d["transcript"]]
        L += [f"- {n}" for n in d["notes"]] + [""]
        if sec["client_choices"]:
            L += ["## Your choices against the recommendations", "", "| Step | Decision | Recommended | Your choice | Note |",
                  "| --- | --- | --- | --- | --- |"]
            L += [f"| {c['step']} | {c['decision']} | {c['recommended']} | {c['client_choice'] or ''} | {c['note'] or ''} |"
                  for c in sec["client_choices"]] + [""]
        if sec.get("importance"):
            imp = sec["importance"]
            if (sec.get("business") or {}).get("signal", True):
                L += [f"## Main drivers ({imp['method'].replace('_', ' ')})", ""]
            else:
                L += [f"## Feature importance ({imp['method'].replace('_', ' ')})", "",
                      "The model does not beat the baseline by more than chance, so these numbers describe noise, "
                      "not what drives the outcome.", ""]
            L += [f"- {k}: {v:.4f}" for k, v in list(imp["importance"].items())[:10]] + [""]
        if sec.get("research"):
            L += ["## Research context", "", sec["research"]["note"], "", f"Novelty: {sec['research']['novelty']}.", ""]
        if sec["memory"].get("suggestions"):
            L += ["## Memory suggestions (advice only)", ""]
            L += [f"- {m['model']} from a dataset at distance {m['distance']:.3f} ({m['created_at'][:10]}): test {m['metric']} "
                  f"{m['test_score']:.3f} vs baseline {m['baseline_score']:.3f}" for m in sec["memory"]["suggestions"]] + [""]
        L += ["## Plots", ""] + [f"- {p['title']}: {p['caption']} (`plots/{p['id']}.json`)" for p in sec["plots"]] + [""]
        L += ["## Limitations", ""] + [f"- {x}" for x in sec["limits"]] + [""]
        L += ["## Code", "", "`pipeline.py` in the export bundle reproduces the held-out score: `python pipeline.py --data <your.csv>`.", ""]
        return "\n".join(L)

    def html(self, sec: dict, plots: list[dict], static: bool = False) -> str:
        """Self-contained HTML. Every string from data or the LLM is escaped; plots render from JSON."""
        md_body = self.markdown(sec)
        body = []
        for line in md_body.splitlines():
            if line.startswith("# "):
                body.append(f"<h1>{_e(line[2:])}</h1>")
            elif line.startswith("## "):
                body.append(f"<h2>{_e(line[3:])}</h2>")
            elif line.startswith("| ---"):
                continue
            elif line.startswith("| "):
                cells = [c.strip() for c in line.strip("|").split(" | ")]
                body.append("<tr>" + "".join(f"<td>{_e(c.replace(chr(92) + '|', '|'))}</td>" for c in cells) + "</tr>")
            elif line.startswith("- "):
                body.append(f"<li>{_e(line[2:])}</li>")
            elif line.strip():
                body.append(f"<p>{_e(line)}</p>")
        out, in_table, in_list = [], False, False
        for b in body:
            if b.startswith("<tr>") and not in_table:
                out.append("<table>")
                in_table = True
            if not b.startswith("<tr>") and in_table:
                out.append("</table>")
                in_table = False
            if b.startswith("<li>") and not in_list:
                out.append("<ul>")
                in_list = True
            if not b.startswith("<li>") and in_list:
                out.append("</ul>")
                in_list = False
            out.append(b.replace("**", ""))
        if in_table:
            out.append("</table>")
        if in_list:
            out.append("</ul>")
        plot_html = ""
        if not static:
            divs = "".join(f'<figure><div id="p{i}" class="plot"></div><figcaption>{_e(p["caption"])}</figcaption></figure>'
                           for i, p in enumerate(plots))
            data = json.dumps([p["figure"] for p in plots]).replace("</", "<\\/")
            plot_html = (f"<h2>Interactive plots</h2>{divs}<script src=\"{PLOTLY_CDN}\"></script>"
                         f"<script>const F={data};F.forEach((f,i)=>Plotly.newPlot('p'+i,f.data,f.layout,{{responsive:true}}));</script>")
        css = ("body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;"
               "color:#1d2433;line-height:1.5}table{border-collapse:collapse;margin:8px 0;font-size:14px}"
               "td{border:1px solid #d5dbe5;padding:4px 8px;vertical-align:top}h1{font-size:26px}h2{margin-top:28px;"
               "border-bottom:1px solid #e3e7ee}.plot{height:360px}figure{margin:16px 0}figcaption{font-size:13px;color:#5b6475}")
        return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
                f"content=\"width=device-width,initial-scale=1\"><title>{_e(sec['title'])}</title><style>{css}</style></head>"
                f"<body>{''.join(out)}{plot_html}</body></html>")
