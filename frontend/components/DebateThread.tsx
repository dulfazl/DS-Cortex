"use client";

import type { EvidencePayload } from "./EvidenceDrawer";
import { AGENT_NAMES } from "@/lib/format";
import type { DebateTurn } from "@/lib/types";

const KIND_STYLE: Record<string, string> = {
  position: "border-border",
  verification: "border-accent",
  conflict: "border-warn",
  resolution: "border-warn",
  rebuttal: "border-accent",
  decision: "border-good",
  note: "border-border",
};

function initials(agent: string) {
  return (AGENT_NAMES[agent] ?? agent)
    .split(" ")
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

export function DebateThread({
  transcript,
  rule,
  forced,
  notes,
  onEvidence,
}: {
  transcript: DebateTurn[];
  rule: string;
  forced: boolean;
  notes: string[];
  onEvidence: (p: EvidencePayload) => void;
}) {
  const rounds = [...new Set(transcript.map((t) => t.round))];
  return (
    <div className="space-y-6">
      <p className="text-sm text-muted">
        Rule applied: <span className="font-semibold text-text">{rule.replace(/_/g, " ")}</span>
        {forced && <span className="ml-2 rounded bg-warn-soft px-1.5 text-xs font-semibold text-warn">forced tiebreak</span>}. The LLM may explain
        a decision; the resolution table makes it.
      </p>
      {rounds.map((r) => (
        <div key={r}>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">Round {r}</p>
          <ol className="space-y-2">
            {transcript
              .filter((t) => t.round === r)
              .map((t, i) => {
                const verified = t.claims.filter((c) => c.verified).length;
                const rejected = t.claims.filter((c) => c.verified === false).length;
                return (
                  <li key={i} className={`flex gap-3 rounded-lg border-l-4 bg-surface p-3 ${KIND_STYLE[t.kind] ?? "border-border"}`}>
                    <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-surface-2 text-xs font-bold" aria-hidden>
                      {initials(t.agent)}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm">
                        <span className="font-semibold">{AGENT_NAMES[t.agent] ?? t.agent}</span>
                        <span className="ml-2 text-xs text-muted">{t.kind}</span>
                      </p>
                      <p className="mt-0.5 text-sm">{t.text}</p>
                      {t.claims.length > 0 && (
                        <button
                          className="mt-1 text-xs text-accent hover:underline"
                          onClick={() =>
                            onEvidence({
                              title: `${AGENT_NAMES[t.agent] ?? t.agent}, round ${t.round}`,
                              subtitle: t.text,
                              evidence: t.claims.flatMap((c) => c.evidence),
                              claims: t.claims,
                            })
                          }
                        >
                          {t.claims.length} claim{t.claims.length > 1 ? "s" : ""}: {verified} verified
                          {rejected ? `, ${rejected} rejected` : ""}. Show evidence
                        </button>
                      )}
                    </div>
                  </li>
                );
              })}
          </ol>
        </div>
      ))}
      {notes.length > 0 && (
        <div className="rounded-lg bg-surface-2 p-3 text-sm">
          {notes.map((n, i) => (
            <p key={i}>{n}</p>
          ))}
        </div>
      )}
    </div>
  );
}
