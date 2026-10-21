"use client";

import { useEffect, useState } from "react";
import type { Claim, Evidence } from "@/lib/types";

export interface EvidencePayload {
  title: string;
  subtitle?: string;
  evidence: Evidence[];
  claims?: Claim[];
}

function Json({ value }: { value: unknown }) {
  const text = JSON.stringify(value, null, 2);
  const [open, setOpen] = useState(text.length < 1200);
  return (
    <div>
      <pre className={`overflow-auto rounded-md bg-surface-2 p-2 font-mono text-[11px] leading-relaxed ${open ? "max-h-96" : "max-h-24"}`}>{text}</pre>
      {text.length >= 1200 && (
        <button className="mt-1 text-xs text-accent" onClick={() => setOpen(!open)}>
          {open ? "Collapse" : "Show the full result"}
        </button>
      )}
    </div>
  );
}

/** Expands the executed tool results behind a claim or a confidence badge. */
export function EvidenceDrawer({ payload, onClose }: { payload: EvidencePayload | null; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  if (!payload) return null;
  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/30" onClick={onClose} role="dialog" aria-modal="true" aria-label="Evidence">
      <aside className="h-full w-full max-w-xl overflow-y-auto bg-surface p-5 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="mb-4 flex items-start gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-muted">Evidence</p>
            <h3 className="text-lg font-semibold">{payload.title}</h3>
            {payload.subtitle && <p className="mt-1 text-sm text-muted">{payload.subtitle}</p>}
          </div>
          <button className="btn btn-ghost ml-auto" onClick={onClose} aria-label="Close">
            Close
          </button>
        </div>
        {payload.claims && payload.claims.length > 0 && (
          <div className="mb-5">
            <h4 className="mb-2 text-sm font-semibold">Claims</h4>
            <ul className="space-y-2">
              {payload.claims.map((c, i) => (
                <li key={i} className="rounded-lg border border-border p-2 text-sm">
                  <span className={`mr-2 font-semibold ${c.verified === false ? "text-bad" : c.verified ? "text-good" : "text-muted"}`}>
                    {c.verified === false ? "✗ rejected" : c.verified ? "✓ verified" : "not checked"}
                  </span>
                  {c.text}
                  {c.source === "llm" && <span className="ml-2 rounded bg-surface-2 px-1 text-[11px] text-muted">LLM</span>}
                  {c.verify_note && <p className="mt-1 text-xs text-bad">{c.verify_note}</p>}
                  {c.checks.map((k, j) => (
                    <p key={j} className="mt-1 font-mono text-[11px] text-muted">
                      check: {k.path} {k.op} {k.value === null || k.value === undefined ? "" : JSON.stringify(k.value)} · run {k.run_id.slice(0, 8)}
                    </p>
                  ))}
                </li>
              ))}
            </ul>
          </div>
        )}
        <h4 className="mb-2 text-sm font-semibold">Tool runs ({payload.evidence.length})</h4>
        <div className="space-y-3">
          {payload.evidence.map((e, i) => (
            <div key={`${e.run_id}-${i}`} className="rounded-lg border border-border p-3">
              <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
                <span className="font-mono font-semibold">{e.tool}</span>
                <span className="rounded bg-surface-2 px-1.5 font-mono text-[11px] text-muted">run {e.run_id.slice(0, 12)}</span>
              </div>
              {Object.keys(e.inputs ?? {}).length > 0 && (
                <>
                  <p className="mb-1 text-xs text-muted">Inputs</p>
                  <Json value={e.inputs} />
                </>
              )}
              <p className="mb-1 mt-2 text-xs text-muted">Executed result</p>
              <Json value={e.result} />
            </div>
          ))}
          {payload.evidence.length === 0 && <p className="text-sm text-muted">No tool runs attached.</p>}
        </div>
      </aside>
    </div>
  );
}
