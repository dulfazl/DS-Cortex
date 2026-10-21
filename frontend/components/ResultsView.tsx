"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { fmt, label, LOWER_IS_BETTER } from "@/lib/format";
import type { Results } from "@/lib/types";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { DebateThread } from "./DebateThread";
import { DriftPanel } from "./DriftPanel";
import { EvidenceDrawer, type EvidencePayload } from "./EvidenceDrawer";
import { LedgerTable } from "./LedgerTable";
import { PlotCard } from "./PlotCard";
import { Banner, Card, SeverityPill } from "./ui";

const TABS = ["Overview", "Plots", "Model", "Decision Ledger", "Debate Transcript", "Code", "Drift"] as const;
type Tab = (typeof TABS)[number];

function Kpi({ title, value, sub, tone }: { title: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="rounded-xl bg-surface-2 px-4 py-3">
      <p className="text-xs text-muted">{title}</p>
      <p className={`mt-0.5 text-2xl font-semibold tabular-nums ${tone ?? ""}`}>{value}</p>
      {sub && <p className="text-xs text-muted">{sub}</p>}
    </div>
  );
}

function Overview({ r, onEvidence }: { r: Results; onEvidence: (p: EvidencePayload) => void }) {
  const t = r.test;
  const better = LOWER_IS_BETTER.has(t.metric) ? t.score < t.baseline.score : t.score > t.baseline.score;
  const final = r.ledger.find((e) => e.decision_point === "final_pipeline");
  const choice = r.ledger.find((e) => e.decision_point === "model_choice");
  const b = r.explain.business;
  const [depth, setDepth] = useState<"technical" | "beginner">("technical");
  const narrative = depth === "technical" ? r.explain.narrative : r.explain.narrative_beginner;
  return (
    <div className="space-y-4">
      {r.override && (
        <Banner tone="warn" title={`You chose ${r.override.client} over the recommended ${r.override.recommended}`}>
          {r.override.warning ?? "Within noise of the recommendation."}
          {t.shadow && (
            <p className="mt-1">
              Shadow run on the held-out rows: yours {fmt(t.shadow.client_score, 4)}, recommended {fmt(t.shadow.recommended_score, 4)}.
            </p>
          )}
        </Banner>
      )}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi title={`Held-out ${t.metric}`} value={fmt(t.score, 4)} sub={`${t.n_test} rows never used for training or selection`} tone={better ? "text-good" : "text-bad"} />
        <Kpi title="Baseline" value={fmt(t.baseline.score, 4)} sub={LOWER_IS_BETTER.has(t.metric) ? "lower is better" : "higher is better"} />
        <Kpi title="Runner-up" value={t.runner_up ? fmt(t.runner_up.score, 4) : "–"} sub={t.runner_up?.id} />
        <div className="rounded-xl bg-surface-2 px-4 py-3">
          <p className="text-xs text-muted">Confidence in the choice</p>
          <div className="mt-1">
            <ConfidenceBadge
              value={r.final.confidence.confidence}
              method="bootstrap"
              onClick={() =>
                final &&
                onEvidence({
                  title: "Confidence in the final pipeline",
                  subtitle: `${r.final.confidence.formula}: stability ${fmt(r.final.confidence.stability, 2)}, separation ${fmt(
                    r.final.confidence.separation,
                    2,
                  )}, effect ${fmt(r.final.confidence.effect, 2)}`,
                  evidence: final.recommendation.evidence,
                  claims: choice?.recommendation.claims,
                })
              }
            />
          </div>
          {r.calibration && (
            <p className="mt-1 text-xs text-muted">Held up on the test set: {r.calibration.outcome ? "yes" : "no"}</p>
          )}
        </div>
      </div>
      <Card title={`Chosen pipeline: ${r.final.chosen}`}>
        <ol className="list-decimal space-y-0.5 pl-5 text-sm">
          {r.candidates.find((c) => c.id === r.final.chosen)?.steps.map((s, i) => <li key={i}>{s}</li>)}
        </ol>
        <p className="mt-2 text-sm text-muted">{r.final.note}.</p>
      </Card>
      {b && (
        <Card title="Summary for stakeholders">
          <p className="text-sm">{b.headline}</p>
          {b.recommendations.length > 0 && (
            <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
              {b.recommendations.map((x, i) => (
                <li key={i}>
                  {x.text} <span className="text-xs text-muted">(source: {x.cites})</span>
                </li>
              ))}
            </ul>
          )}
          {b.caveats.map((c, i) => (
            <p key={i} className="mt-2 text-sm text-warn">
              Caveat: {c}
            </p>
          ))}
        </Card>
      )}
      <Card
        title="What was done and why"
        actions={
          <div className="flex rounded-lg border border-border p-0.5 text-xs">
            {(["technical", "beginner"] as const).map((d) => (
              <button key={d} onClick={() => setDepth(d)} className={`rounded-md px-2 py-1 ${depth === d ? "bg-accent text-white" : "text-muted"}`}>
                {d}
              </button>
            ))}
          </div>
        }
      >
        <ol className="space-y-1.5 text-sm">
          {narrative?.items.map((i) => (
            <li key={i.step}>
              <span className="mr-2 font-mono text-xs text-muted">step {i.step}</span>
              {i.text}
            </li>
          ))}
        </ol>
      </Card>
      <Card title="Data audit">
        <p className="mb-2 text-sm text-muted">{r.audit.summary}</p>
        <ul className="space-y-1.5">
          {r.audit.findings.slice(0, 12).map((f, i) => (
            <li key={i} className="flex flex-wrap items-start gap-2 text-sm">
              <SeverityPill severity={f.severity} />
              <span className="font-medium">{label(f.issue_type)}</span>
              {f.column && <span className="font-mono text-xs">{f.column}</span>}
              <span className="min-w-0 flex-1 text-muted">{f.rationale}</span>
              <ConfidenceBadge
                value={f.confidence}
                method={f.confidence_method}
                onClick={() => onEvidence({ title: `${label(f.issue_type)} ${f.column ?? ""}`, subtitle: f.rationale, evidence: f.evidence })}
              />
            </li>
          ))}
        </ul>
      </Card>
      {r.stats.guidance.length > 0 && (
        <Card title="Statistical guidance">
          <ul className="list-disc space-y-1 pl-5 text-sm">
            {r.stats.guidance.map((g, i) => (
              <li key={i}>{g}</li>
            ))}
          </ul>
        </Card>
      )}
      {r.memory.suggestions && r.memory.suggestions.length > 0 && (
        <Card title="Memory suggestions (advice only)">
          <ul className="space-y-1 text-sm">
            {r.memory.suggestions.map((m, i) => (
              <li key={i}>
                <span className="font-medium">{m.model}</span> from a dataset at distance {fmt(m.distance, 3)} ({m.created_at.slice(0, 10)}): test {m.metric}{" "}
                {fmt(m.test_score, 3)} against baseline {fmt(m.baseline_score, 3)}
              </li>
            ))}
          </ul>
        </Card>
      )}
      {r.explain.research && (
        <Card title="Research context">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted">Context, not a literature review</p>
          <p className="mt-1 text-sm">{r.explain.research.note.replace("Context, not a literature review. ", "")}</p>
          <p className="mt-1 text-sm text-muted">Novelty: {r.explain.research.novelty}.</p>
        </Card>
      )}
    </div>
  );
}

function ModelTab({ r, onEvidence }: { r: Results; onEvidence: (p: EvidencePayload) => void }) {
  const imp = r.explain.importance;
  const sorted = [...r.candidates].sort((a, b) => {
    const s = LOWER_IS_BETTER.has(r.plan.metric) ? 1 : -1;
    return s * ((a.cv?.primary_mean ?? 0) - (b.cv?.primary_mean ?? 0));
  });
  return (
    <div className="space-y-4">
      <Card title={`Candidates (cross-validated ${r.plan.metric})`}>
        <div className="overflow-x-auto">
          <table className="tbl">
            <thead>
              <tr>
                <th>Candidate</th>
                <th>Origin</th>
                <th>Status</th>
                <th>CV mean ± sd</th>
                <th>Critic objections</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((c) => (
                <tr key={c.id} className={c.id === r.final.chosen ? "bg-good-soft" : undefined}>
                  <td>
                    <div className="font-mono text-xs font-semibold">{c.id}</div>
                    <div className="text-xs text-muted">{c.rationale.split(" | ")[0]}</div>
                  </td>
                  <td className="text-xs">{c.origin}</td>
                  <td className="text-xs">{c.status}</td>
                  <td className="whitespace-nowrap tabular-nums">
                    {fmt(c.cv?.primary_mean, 4)} <span className="text-muted">± {fmt(c.cv?.primary_std, 4)}</span>
                  </td>
                  <td>
                    {c.objections.length === 0 && <span className="text-xs text-muted">none</span>}
                    {c.objections.map((o, i) => (
                      <button
                        key={i}
                        onClick={() => onEvidence({ title: `${c.id}: ${label(o.check)}`, subtitle: o.message, evidence: o.claim.evidence, claims: [o.claim] })}
                        className={`mb-1 block text-left text-xs hover:underline ${o.resolved ? "text-muted line-through" : ""}`}
                      >
                        <SeverityPill severity={o.severity} /> {o.message}
                      </button>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      {imp && (
        <Card title={`Main drivers (${label(imp.method)})`}>
          {imp.substituted && <Banner tone="warn" title="SHAP was not available">{imp.substitution_reason}. Permutation importance is shown instead.</Banner>}
          <ul className="mt-2 space-y-1">
            {Object.entries(imp.importance)
              .slice(0, 12)
              .map(([k, v]) => {
                const max = Math.max(...Object.values(imp.importance).map(Math.abs)) || 1;
                return (
                  <li key={k} className="flex items-center gap-2 text-sm">
                    <span className="w-40 shrink-0 truncate" title={k}>
                      {k}
                    </span>
                    <span className="h-2 rounded bg-accent" style={{ width: `${Math.max(2, (Math.abs(v) / max) * 60)}%` }} />
                    <span className="font-mono text-xs text-muted">{fmt(v, 4)}</span>
                  </li>
                );
              })}
          </ul>
        </Card>
      )}
    </div>
  );
}

function CodeTab({ id, files }: { id: string; files: string[] }) {
  const [code, setCode] = useState<string | null>(null);
  useEffect(() => {
    fetch(api.exportUrl(id, "pipeline.py"))
      .then((r) => (r.ok ? r.text() : Promise.reject(r.statusText)))
      .then(setCode)
      .catch(() => setCode(null));
  }, [id]);
  const downloads: [string, string][] = [
    ["Everything (zip)", ""],
    ["Report (HTML)", "report.html"],
    ["Report (Markdown)", "report.md"],
    ["Report (PDF)", "report.pdf"],
    ["Pipeline script", "pipeline.py"],
    ["Decision ledger (JSON)", "ledger.json"],
    ["Model card", "model/model_card.md"],
  ];
  return (
    <div className="space-y-4">
      <Card title="Downloads">
        <div className="flex flex-wrap gap-2">
          {downloads
            .filter(([, f]) => !f || files.includes(f))
            .map(([t, f]) => (
              <a key={t} className="btn btn-ghost" href={api.exportUrl(id, f || undefined)} download>
                {t}
              </a>
            ))}
        </div>
      </Card>
      <Card title="pipeline.py: reproduces the held-out score">
        <p className="mb-2 text-sm text-muted">
          Run <code className="rounded bg-surface-2 px-1 font-mono">python pipeline.py --data your.csv</code>. It inlines the exact loader and pipeline code
          this run used.
        </p>
        <pre className="max-h-[480px] overflow-auto rounded-lg bg-surface-2 p-3 font-mono text-xs">{code ?? "Loading…"}</pre>
      </Card>
    </div>
  );
}

export function ResultsView({ results, sessionId }: { results: Results; sessionId: string | null }) {
  const [tab, setTab] = useState<Tab>("Overview");
  const [evidence, setEvidence] = useState<EvidencePayload | null>(null);
  const r = results;
  const tabs = TABS.filter((t) => sessionId || (t !== "Code" && t !== "Drift"));
  return (
    <div>
      <div className="mb-4 flex gap-1 overflow-x-auto border-b border-border" role="tablist">
        {tabs.map((t) => (
          <button
            key={t}
            role="tab"
            aria-selected={tab === t}
            onClick={() => setTab(t)}
            className={`whitespace-nowrap border-b-2 px-3 py-2 text-sm ${tab === t ? "border-accent font-semibold text-text" : "border-transparent text-muted hover:text-text"}`}
          >
            {t}
          </button>
        ))}
      </div>
      {tab === "Overview" && <Overview r={r} onEvidence={setEvidence} />}
      {tab === "Plots" && (
        <div className="grid gap-4 md:grid-cols-2">
          {r.plots.map((p) => (
            <PlotCard key={p.id} plot={p} />
          ))}
        </div>
      )}
      {tab === "Model" && <ModelTab r={r} onEvidence={setEvidence} />}
      {tab === "Decision Ledger" && (
        <Card
          title={`Decision Ledger (${r.ledger.length} entries)`}
          actions={
            sessionId && (
              <a className="btn btn-ghost" href={api.exportUrl(sessionId, "ledger.json")} download>
                Export JSON
              </a>
            )
          }
        >
          <LedgerTable entries={r.ledger} onEvidence={setEvidence} />
        </Card>
      )}
      {tab === "Debate Transcript" &&
        (r.debate ? (
          <Card title="Debate transcript">
            <DebateThread transcript={r.debate.transcript} rule={r.debate.rule} forced={r.debate.forced_tiebreak} notes={r.debate.notes} onEvidence={setEvidence} />
          </Card>
        ) : (
          <p className="text-sm text-muted">No debate was held for this run.</p>
        ))}
      {tab === "Code" && sessionId && <CodeTab id={sessionId} files={r.export.files} />}
      {tab === "Drift" && sessionId && <DriftPanel id={sessionId} />}
      <EvidenceDrawer payload={evidence} onClose={() => setEvidence(null)} />
    </div>
  );
}
