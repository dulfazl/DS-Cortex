"use client";

import type { ReactNode } from "react";
import type { Status } from "@/lib/types";

const STATUS_STYLE: Record<string, string> = {
  created: "bg-surface-2 text-muted",
  queued: "bg-warn-soft text-warn",
  running: "bg-accent-soft text-accent",
  awaiting_approval: "bg-warn-soft text-warn",
  awaiting_clarification: "bg-warn-soft text-warn",
  blocked: "bg-bad-soft text-bad",
  completed: "bg-good-soft text-good",
  failed: "bg-bad-soft text-bad",
  interrupted: "bg-warn-soft text-warn",
};

const STATUS_LABEL: Record<string, string> = {
  awaiting_approval: "waiting for your approval",
  awaiting_clarification: "waiting for your answer",
  blocked: "blocked: needs your decision",
};

export function StatusPill({ status }: { status: Status | string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold ${STATUS_STYLE[status] ?? ""}`}>
      {status === "running" && <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-accent" />}
      {STATUS_LABEL[status] ?? status}
    </span>
  );
}

export function SeverityPill({ severity }: { severity: string }) {
  const s =
    severity === "blocker" || severity === "high"
      ? "bg-bad-soft text-bad"
      : severity === "medium"
        ? "bg-warn-soft text-warn"
        : "bg-surface-2 text-muted";
  return <span className={`rounded px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${s}`}>{severity}</span>;
}

export function Card({ title, actions, children, className = "" }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card p-4 ${className}`}>
      {(title || actions) && (
        <div className="mb-3 flex items-center gap-2">
          {title && <h2 className="text-sm font-semibold">{title}</h2>}
          <div className="ml-auto flex items-center gap-2">{actions}</div>
        </div>
      )}
      {children}
    </section>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} />;
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="grid place-items-center rounded-xl border border-dashed border-border px-6 py-12 text-center">
      <p className="font-semibold">{title}</p>
      {children && <div className="mt-1 max-w-md text-sm text-muted">{children}</div>}
    </div>
  );
}

export function Banner({ tone, title, children }: { tone: "bad" | "warn" | "good" | "info"; title: string; children?: ReactNode }) {
  const style = {
    bad: "border-bad bg-bad-soft",
    warn: "border-warn bg-warn-soft",
    good: "border-good bg-good-soft",
    info: "border-accent bg-accent-soft",
  }[tone];
  return (
    <div role={tone === "bad" ? "alert" : "status"} className={`rounded-xl border-l-4 px-4 py-3 ${style}`}>
      <p className="text-sm font-semibold">{title}</p>
      {children && <div className="mt-1 text-sm">{children}</div>}
    </div>
  );
}
