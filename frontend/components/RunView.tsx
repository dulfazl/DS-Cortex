"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";
import { useRunStore } from "@/lib/store";
import type { LedgerEntry, Results, RunEvent, SessionSummary } from "@/lib/types";
import { useEventStream } from "@/lib/useEventStream";
import { AgentTimeline } from "./AgentTimeline";
import { ChatPanel } from "./ChatPanel";
import { EventFeed } from "./EventFeed";
import { EvidenceDrawer, type EvidencePayload } from "./EvidenceDrawer";
import { LedgerTable } from "./LedgerTable";
import { ApprovalPanel, BlockerPanel, ClarifyCard } from "./PausePanels";
import { QuickProfile } from "./QuickProfile";
import { ResultsView } from "./ResultsView";
import { Banner, Card, Skeleton, StatusPill } from "./ui";

const REFRESH_ON = new Set(["stage", "approval_required", "clarification_required", "blocker", "completed", "failed", "status", "queued", "warning"]);

async function loadRun(id: string): Promise<{ s: SessionSummary; results: Results | null; partial: LedgerEntry[] | null }> {
  const s = await api.session(id);
  const results = s.status === "completed" ? await api.results(id) : null;
  const partial = s.status === "failed" || s.status === "interrupted" ? await api.ledger(id) : null;
  return { s, results, partial };
}

export function RunView({ id }: { id: string }) {
  const router = useRouter();
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [lost, setLost] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [showTools, setShowTools] = useState(false);
  const [mode, setMode] = useState<"autopilot" | "review">("autopilot");
  const [starting, setStarting] = useState(false);
  const [partial, setPartial] = useState<LedgerEntry[] | null>(null);
  const [evidence, setEvidence] = useState<EvidencePayload | null>(null);
  const events = useRunStore((s) => s.events);
  const connected = useRunStore((s) => s.connected);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(
    () =>
      loadRun(id)
        .then(({ s, results: r, partial: led }) => {
          setSession(s);
          setMode(s.mode);
          if (r) setResults(r);
          if (led) setPartial(led);
        })
        .catch((e) => {
          // 404 after the run was loaded: the server restarted without a disk and the run is gone
          if (e instanceof ApiError && e.status === 404) setLost(true);
          else setLoadError(e instanceof Error ? e.message : String(e));
        }),
    [id],
  );

  useEffect(() => {
    let cancelled = false;
    loadRun(id)
      .then(({ s, results: r, partial: led }) => {
        if (cancelled) return;
        setSession(s);
        setMode(s.mode);
        if (r) setResults(r);
        if (led) setPartial(led);
      })
      .catch((e) => !cancelled && setLoadError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [id]);

  const onEvent = useCallback(
    (e: RunEvent) => {
      if (e.type === "warning") setWarning(String(e.message));
      if (REFRESH_ON.has(e.type)) {
        if (timer.current) clearTimeout(timer.current);
        timer.current = setTimeout(refresh, 250);
      }
    },
    [refresh],
  );
  const onDrop = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(refresh, 250);
  }, [refresh]);
  useEventStream(id, { onEvent, onDrop, enabled: !!session && session.status !== "completed" && !lost });

  // While the run works, re-read it every minute: this catches a status change the stream missed, and on a
  // free host that sleeps after 15 minutes without incoming requests it keeps the server awake until the run ends.
  const working = !!session && ["running", "queued"].includes(session.status) && !lost;
  useEffect(() => {
    if (!working) return;
    const t = setInterval(refresh, 60_000);
    return () => clearInterval(t);
  }, [working, refresh]);

  const start = async () => {
    setStarting(true);
    try {
      await api.start(id, mode);
      await refresh();
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };

  const rerun = async () => {
    const r = await api.rerun(id);
    router.push(`/sessions/${r.session_id}`);
  };

  const afterAnswer = (w?: string | null) => {
    if (w) setWarning(w);
    refresh();
  };

  if (loadError && !session) return <Banner tone="bad" title="Could not load this run">{loadError}</Banner>;
  if (!session)
    return (
      <div className="space-y-3">
        <Skeleton className="h-10 w-1/2" />
        <Skeleton className="h-64 w-full" />
      </div>
    );

  if (lost)
    return (
      <Banner tone="bad" title="This run is no longer on the server">
        The server restarted while “{session.file_name}” was being analysed. On the free hosting plan the server has no disk, so a
        restart erases runs in progress. Upload the file again to start a new run.
        <p className="mt-2">
          <button className="btn btn-primary" onClick={() => router.push("/")}>
            New run
          </button>
        </p>
      </Banner>
    );

  const s = session;
  const p = s.pending;
  const live = !["created", "completed"].includes(s.status);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="truncate text-xl font-semibold">{s.file_name}</h1>
            <StatusPill status={s.status} />
            <span className="rounded bg-surface-2 px-1.5 text-xs text-muted">{s.mode}</span>
            {live && <span className={`text-xs ${connected ? "text-good" : "text-muted"}`}>{connected ? "● live" : "○ reconnecting"}</span>}
          </div>
          <p className="mt-1 text-sm text-muted">“{s.problem_statement}”</p>
          {s.task_plan && (
            <p className="mt-1 text-xs text-muted">
              {s.task_plan.task_family.replace("_", " ")} · target {s.task_plan.target ?? "none"} · metric {s.task_plan.metric} (fixed before
              modelling)
            </p>
          )}
        </div>
        <div className="ml-auto flex flex-wrap gap-2">
          {s.status === "completed" && (
            <button className="btn btn-ghost" onClick={() => setChatOpen(!chatOpen)}>
              {chatOpen ? "Hide chat" : "Ask about this run"}
            </button>
          )}
          {(s.status === "completed" || s.status === "failed") && (
            <button className="btn btn-ghost" onClick={rerun}>
              Re-run
            </button>
          )}
        </div>
      </div>

      {warning && (
        <Banner tone="warn" title="Override warning">
          {warning}
        </Banner>
      )}

      {(s.status === "created" || s.status === "interrupted") && (
        <>
          {s.status === "interrupted" && (
            <Banner tone="warn" title="The server restarted during this run">
              Finished stages are saved. Resume continues from the first unfinished stage.
            </Banner>
          )}
          {s.profile && <QuickProfile profile={s.profile} plan={s.task_plan} />}
          <Card title="Start the analysis">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex rounded-lg border border-border p-0.5 text-sm">
                {(["autopilot", "review"] as const).map((m) => (
                  <button key={m} onClick={() => setMode(m)} className={`rounded-md px-3 py-1 ${mode === m ? "bg-accent text-white" : "text-muted"}`}>
                    {m}
                  </button>
                ))}
              </div>
              <button className="btn btn-primary" onClick={start} disabled={starting}>
                {starting ? "Starting…" : s.status === "interrupted" ? "Resume" : "Start"}
              </button>
              <p className="text-xs text-muted">
                {mode === "review" ? "Pauses at each stage bundle for your approval." : "Runs end to end; stops only for blockers."}
              </p>
            </div>
          </Card>
        </>
      )}

      {live && s.status !== "interrupted" && (
        <div className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
          <Card title="Agents">
            <AgentTimeline session={s} />
          </Card>
          <div className="min-w-0 space-y-4">
            {s.status === "queued" && (
              <Banner tone="warn" title="Queued">
                Other runs are using the server. Yours starts automatically{s.queue_position ? ` (position ${s.queue_position})` : ""}.
              </Banner>
            )}
            {p?.kind === "clarification" && <ClarifyCard id={id} pending={p} onDone={afterAnswer} />}
            {p?.kind === "blocker" && <BlockerPanel id={id} pending={p} onDone={afterAnswer} />}
            {p?.kind === "approval" && <ApprovalPanel id={id} pending={p} metric={s.task_plan?.metric} onDone={afterAnswer} />}
            {s.status === "failed" && (
              <Banner tone="bad" title="The run failed">
                {s.error}
                <p className="mt-1">The decisions made before the failure are kept below. Re-run starts a fresh run with the same file and question.</p>
              </Banner>
            )}
            {s.memory?.suggestions && s.memory.suggestions.length > 0 && (
              <Banner tone="info" title="Memory suggestion (advice only)">
                {s.memory.suggestions.map((m) => `${m.model} worked on a similar dataset (distance ${m.distance.toFixed(2)})`).join("; ")}. The
                Architect may add it; the Critic may reject it.
              </Banner>
            )}
            <Card
              title="Live run"
              actions={
                <label className="flex items-center gap-1.5 text-xs text-muted">
                  <input type="checkbox" checked={showTools} onChange={() => setShowTools(!showTools)} /> show tool calls
                </label>
              }
            >
              <EventFeed events={events} showTools={showTools} />
            </Card>
            {s.status === "failed" && partial && (
              <Card title={`Partial decision ledger (${partial.length} entries)`}>
                <LedgerTable entries={partial} onEvidence={setEvidence} />
              </Card>
            )}
          </div>
        </div>
      )}

      {s.status === "completed" &&
        (results ? (
          <div className={chatOpen ? "grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]" : ""}>
            <div className="min-w-0">
              <ResultsView results={results} sessionId={id} />
            </div>
            {chatOpen && <ChatPanel id={id} onClose={() => setChatOpen(false)} />}
          </div>
        ) : (
          <div className="space-y-3">
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-80 w-full" />
          </div>
        ))}
      <EvidenceDrawer payload={evidence} onClose={() => setEvidence(null)} />
    </div>
  );
}
