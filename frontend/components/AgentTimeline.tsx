"use client";

import { AGENT_NAMES, STAGES } from "@/lib/format";
import type { SessionSummary } from "@/lib/types";

type StepState = "done" | "active" | "paused" | "pending" | "failed" | "skipped";

export function AgentTimeline({ session }: { session: SessionSummary }) {
  const done = new Set(session.completed_stages);
  const review = session.mode === "review";
  const state = (key: string): StepState => {
    if (done.has(key)) return key.startsWith("approve_") && !review ? "skipped" : "done";
    if (session.current_stage === key) {
      if (session.status === "failed") return "failed";
      if (["awaiting_approval", "awaiting_clarification", "blocked"].includes(session.status)) return "paused";
      if (session.status === "running" || session.status === "queued") return "active";
    }
    return "pending";
  };
  const visible = STAGES.filter((s) => review || !s.key.startsWith("approve_"));
  return (
    <ol className="space-y-1" aria-label="Agent timeline">
      {visible.map((s) => {
        const st = state(s.key);
        const dot = {
          done: "bg-good",
          active: "bg-accent pulse-dot",
          paused: "bg-warn",
          failed: "bg-bad",
          pending: "bg-border",
          skipped: "bg-border",
        }[st];
        return (
          <li key={s.key} className={`flex items-center gap-3 rounded-md px-2 py-1.5 ${st === "active" || st === "paused" ? "bg-surface-2" : ""}`}>
            <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${dot}`} aria-hidden />
            <div className="min-w-0">
              <p className={`text-sm ${st === "pending" ? "text-muted" : "font-medium"}`}>{s.label}</p>
              <p className="text-[11px] text-muted">{AGENT_NAMES[s.agent]}</p>
            </div>
            <span className="ml-auto text-[11px] text-muted">
              {st === "active" ? "running" : st === "paused" ? "waiting for you" : st === "failed" ? "failed" : st === "done" ? "done" : ""}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
