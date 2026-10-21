"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { fmt, label } from "@/lib/format";
import type { Pending } from "@/lib/types";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { Banner, SeverityPill } from "./ui";

type Done = (warning?: string | null) => void;

function useSubmit(onDone: Done) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async (fn: () => Promise<{ warning?: string | null } | unknown>) => {
    setBusy(true);
    setError(null);
    try {
      const r = (await fn()) as { warning?: string | null };
      onDone(r?.warning ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return { busy, error, run };
}

/** The router's single clarifying question, answered with one chip. */
export function ClarifyCard({ id, pending, onDone }: { id: string; pending: Pending; onDone: Done }) {
  const { busy, error, run } = useSubmit(onDone);
  const p = pending.payload;
  return (
    <section className="card border-warn p-4" aria-live="assertive">
      <p className="text-xs font-semibold uppercase tracking-wide text-warn">The router needs one answer</p>
      <h2 className="mt-1 text-lg font-semibold">{p.question}</h2>
      <p className="mt-1 text-sm text-muted">Why: {p.reason}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {(p.options ?? []).map((o) => (
          <button key={o} className="btn btn-ghost" disabled={busy} onClick={() => run(() => api.clarify(id, o))}>
            {o}
          </button>
        ))}
      </div>
      {error && <p className="mt-2 text-sm text-bad">{error}</p>}
    </section>
  );
}

interface BlockerFinding {
  column: string;
  rationale: string;
  confidence: number;
  confidence_method: string;
}

/** Leakage blocker: confirm dropping each column, or explain why it is available at prediction time. */
export function BlockerPanel({ id, pending, onDone }: { id: string; pending: Pending; onDone: Done }) {
  const { busy, error, run } = useSubmit(onDone);
  const cols = (pending.payload.blockers as string[]) ?? [];
  const findings = (pending.payload.findings as BlockerFinding[]) ?? [];
  const [answers, setAnswers] = useState<Record<string, { decision: "drop" | "keep"; explanation: string }>>(
    Object.fromEntries(cols.map((c) => [c, { decision: "drop", explanation: "" }])),
  );
  return (
    <section className="card border-bad p-4" aria-live="assertive">
      <Banner tone="bad" title="Possible target leakage: the run is paused">
        These columns predict the target almost perfectly on their own. A model trained on them would look excellent and fail in use.
      </Banner>
      <ul className="mt-4 space-y-3">
        {cols.map((c) => {
          const f = findings.find((x) => x.column === c);
          const a = answers[c];
          return (
            <li key={c} className="rounded-lg border border-border p-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono font-semibold">{c}</span>
                <SeverityPill severity="blocker" />
                {f && <ConfidenceBadge value={f.confidence} method={f.confidence_method} />}
              </div>
              {f && <p className="mt-1 text-sm text-muted">{f.rationale}</p>}
              <div className="mt-2 flex flex-wrap gap-4 text-sm">
                {(["drop", "keep"] as const).map((d) => (
                  <label key={d} className="flex items-center gap-1.5">
                    <input type="radio" name={`b-${c}`} checked={a.decision === d} onChange={() => setAnswers({ ...answers, [c]: { ...a, decision: d } })} />
                    {d === "drop" ? "Drop the column" : "Keep it: it is available at prediction time"}
                  </label>
                ))}
              </div>
              {a.decision === "keep" && (
                <textarea
                  className="mt-2 w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm"
                  placeholder="Explain why this column is known before the outcome (required, logged in the ledger)"
                  value={a.explanation}
                  onChange={(e) => setAnswers({ ...answers, [c]: { ...a, explanation: e.target.value } })}
                />
              )}
            </li>
          );
        })}
      </ul>
      {error && <p className="mt-2 text-sm text-bad">{error}</p>}
      <button className="btn btn-primary mt-3" disabled={busy} onClick={() => run(() => api.approve(id, { blockers: answers }))}>
        Confirm and continue
      </button>
    </section>
  );
}

interface Action {
  action: string;
  column: string | null;
  reason: string;
  stage: string;
  enabled: boolean;
}

interface CandRow {
  id: string;
  model: string;
  cv_mean: number;
  cv_std: number;
  objections: string[];
  status: string;
}

/** Review mode: one stage bundle at a time, with Approve or Change and a drill-down into each choice. */
export function ApprovalPanel({ id, pending, metric, onDone }: { id: string; pending: Pending; metric?: string; onDone: Done }) {
  const { busy, error, run } = useSubmit(onDone);
  const p = pending.payload as Record<string, unknown>;
  const bundle = pending.key;
  const [disabled, setDisabled] = useState<Set<number>>(new Set());
  const [options, setOptions] = useState<Record<string, boolean>>((p.options as Record<string, boolean>) ?? {});
  const [choice, setChoice] = useState<string>(String(p.recommended ?? ""));
  const [justification, setJustification] = useState("");

  const approve = () => run(() => api.approve(id, { bundle, decision: "approve" }));
  let body: React.ReactNode = null;
  let change: (() => void) | null = null;

  if (bundle === "preprocessing") {
    const actions = (p.actions as Action[]) ?? [];
    body = (
      <>
        <p className="text-sm text-muted">{String(p.summary ?? "")} Untick an action to skip it. Pipeline actions are fitted inside each CV fold.</p>
        <ul className="mt-2 max-h-72 space-y-1 overflow-auto">
          {actions.map((a, i) => (
            <li key={i} className="flex items-start gap-2 rounded-md px-2 py-1 hover:bg-surface-2">
              <input
                type="checkbox"
                className="mt-1"
                checked={a.enabled && !disabled.has(i)}
                disabled={!a.enabled}
                onChange={() => {
                  const d = new Set(disabled);
                  if (d.has(i)) d.delete(i);
                  else d.add(i);
                  setDisabled(d);
                }}
                aria-label={`${a.action} ${a.column ?? ""}`}
              />
              <div className="text-sm">
                <span className="font-medium">{label(a.action)}</span> {a.column && <span className="font-mono text-xs">{a.column}</span>}
                <span className="ml-2 rounded bg-surface-2 px-1 text-[11px] text-muted">{a.stage}</span>
                <p className="text-xs text-muted">{a.reason}</p>
              </div>
            </li>
          ))}
        </ul>
      </>
    );
    change = () => run(() => api.approve(id, { bundle, decision: "change", disabled: [...disabled] }));
  } else if (bundle === "feature_engineering") {
    body = (
      <>
        <ul className="space-y-1 text-sm">
          {Object.entries(options).map(([k, v]) => (
            <li key={k}>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={v} onChange={() => setOptions({ ...options, [k]: !v })} />
                {label(k)}
              </label>
            </li>
          ))}
        </ul>
        {Array.isArray(p.guidance) && (
          <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-muted">
            {(p.guidance as string[]).map((g, i) => (
              <li key={i}>{g}</li>
            ))}
          </ul>
        )}
      </>
    );
    change = () => run(() => api.approve(id, { bundle, decision: "change", options }));
  } else if (bundle === "modelling") {
    const cands = (p.candidates as CandRow[]) ?? [];
    body = (
      <>
        <p className="text-sm">
          Recommended: <span className="font-semibold">{String(p.recommended)}</span>{" "}
          <ConfidenceBadge value={Number(p.confidence)} method="bootstrap" /> by rule <span className="font-mono text-xs">{String(p.rule)}</span>
        </p>
        <p className="mt-1 text-sm text-muted">{String(p.rationale ?? "")}</p>
        <div className="mt-3 overflow-x-auto">
          <table className="tbl">
            <thead>
              <tr>
                <th />
                <th>Candidate</th>
                <th>CV {metric}</th>
                <th>Open objections</th>
              </tr>
            </thead>
            <tbody>
              {cands.map((c) => (
                <tr key={c.id}>
                  <td>
                    <input type="radio" name="choice" checked={choice === c.id} onChange={() => setChoice(c.id)} aria-label={c.id} />
                  </td>
                  <td className="font-mono text-xs">{c.id}</td>
                  <td>
                    {fmt(c.cv_mean, 4)} <span className="text-muted">± {fmt(c.cv_std, 4)}</span>
                  </td>
                  <td className="text-xs text-muted">{c.objections.join("; ") || "none"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {choice !== p.recommended && (
          <textarea
            className="mt-2 w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm"
            placeholder={
              p.rule === "validity_vs_performance"
                ? "Justification (required: the recommendation was decided on statistical validity)"
                : "Justification (optional, logged in the ledger)"
            }
            value={justification}
            onChange={(e) => setJustification(e.target.value)}
          />
        )}
      </>
    );
    change = choice !== p.recommended ? () => run(() => api.approve(id, { bundle, decision: "change", choice, justification })) : null;
  } else if (bundle === "tuning") {
    const d = p.default as { id: string; cv: { primary_mean: number; primary_std: number } };
    const t = p.tuned as { id: string; cv: { primary_mean: number; primary_std: number } };
    body = (
      <div className="text-sm">
        <p>{String(p.note ?? "")}</p>
        <table className="tbl mt-2">
          <tbody>
            <tr>
              <td>Default {d?.id}</td>
              <td>
                {fmt(d?.cv.primary_mean, 4)} ± {fmt(d?.cv.primary_std, 4)}
              </td>
            </tr>
            <tr>
              <td>Proposed {t?.id}</td>
              <td>
                {fmt(t?.cv.primary_mean, 4)} ± {fmt(t?.cv.primary_std, 4)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    );
    change = () => run(() => api.approve(id, { bundle, decision: "change" }));
  }

  return (
    <section className="card border-warn p-4" aria-live="assertive">
      <p className="text-xs font-semibold uppercase tracking-wide text-warn">Review mode · waiting for your approval</p>
      <h2 className="mt-1 text-lg font-semibold">{label(bundle)} bundle</h2>
      <div className="mt-3">{body}</div>
      {error && <p className="mt-2 text-sm text-bad">{error}</p>}
      <div className="mt-4 flex flex-wrap gap-2">
        <button className="btn btn-primary" disabled={busy} onClick={approve}>
          Approve
        </button>
        {change && (
          <button className="btn btn-ghost" disabled={busy} onClick={change}>
            {bundle === "modelling" ? "Use my choice" : bundle === "tuning" ? "Keep the default" : "Apply my changes"}
          </button>
        )}
      </div>
    </section>
  );
}
