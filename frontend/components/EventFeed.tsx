"use client";

import { AGENT_NAMES, label } from "@/lib/format";
import type { RunEvent } from "@/lib/types";
import { ConfidenceBadge } from "./ConfidenceBadge";

function line(e: RunEvent): { text: string; tone?: string; badge?: { v: number; m: string } } | null {
  const agent = AGENT_NAMES[String(e.agent ?? "")] ?? String(e.agent ?? "");
  switch (e.type) {
    case "agent_started":
      return { text: `${agent} started${e.detail ? `: ${e.detail}` : ""}` };
    case "agent_finished":
      return { text: `${agent} finished${e.detail ? `: ${e.detail}` : ""}`, tone: "text-good" };
    case "ledger_entry":
      return {
        text: `Ledger step ${e.step}: ${label(String(e.decision_point))} → ${e.chosen}`,
        badge: { v: Number(e.confidence), m: String(e.method) },
      };
    case "candidate_evaluated":
      return { text: `Evaluated ${e.id}: ${e.metric} ${Number(e.mean).toFixed(4)} ± ${Number(e.std).toFixed(4)}` };
    case "critic_review":
      return { text: `Critic on ${e.candidate}: ${(e.objections as string[]).length ? (e.objections as string[]).join(", ") : "no objections"}` };
    case "critic_round":
      return { text: `Revision round ${e.round}: ${(e.changes as string[]).join("; ")}` };
    case "memory_suggestion":
      return { text: `Memory: ${(e.suggestions as unknown[]).length} similar past dataset(s) suggested a pipeline (advice only)`, tone: "text-accent" };
    case "plot_ready":
      return { text: `Plot ready: ${e.title}` };
    case "approval_required":
      return { text: `Waiting for your approval: ${label(String(e.key))}`, tone: "text-warn" };
    case "clarification_required":
      return { text: "Waiting for your answer to a clarifying question", tone: "text-warn" };
    case "blocker":
      return { text: `Blocker: possible target leakage in ${(e.blockers as string[]).join(", ")}`, tone: "text-bad" };
    case "warning":
      return { text: String(e.message), tone: "text-warn" };
    case "queued":
      return { text: String(e.message ?? "Queued"), tone: "text-warn" };
    case "failed":
      return { text: `Run failed: ${e.message}`, tone: "text-bad" };
    case "completed":
      return { text: "Run complete", tone: "text-good" };
    default:
      return null;
  }
}

export function EventFeed({ events, showTools }: { events: RunEvent[]; showTools: boolean }) {
  const rows = events
    .filter((e) => showTools || e.type !== "tool_called")
    .map((e) => ({ e, l: e.type === "tool_called" ? { text: `tool ${e.tool} (${Number(e.seconds).toFixed(2)} s)`, tone: "text-muted" } : line(e) }))
    .filter((r) => r.l);
  if (rows.length === 0) return <p className="text-sm text-muted">Waiting for the first event…</p>;
  return (
    <ol className="max-h-[520px] space-y-1 overflow-y-auto pr-1 font-mono text-xs" aria-live="polite">
      {rows
        .slice(-300)
        .reverse()
        .map(({ e, l }) => (
          <li key={e.seq} className="flex items-start gap-2">
            <span className="w-10 shrink-0 text-right text-muted">{e.seq}</span>
            <span className={`min-w-0 flex-1 break-words ${l!.tone ?? ""}`}>{l!.text}</span>
            {l!.badge && <ConfidenceBadge value={l!.badge.v} method={l!.badge.m} />}
          </li>
        ))}
    </ol>
  );
}
