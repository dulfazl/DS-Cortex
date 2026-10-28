"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api, DEMO, type ReplayInfo } from "@/lib/api";
import { fmt } from "@/lib/format";
import { useRunStore } from "@/lib/store";
import type { Results, RunEvent } from "@/lib/types";
import { useEventStream } from "@/lib/useEventStream";
import { EventFeed } from "./EventFeed";
import { ResultsView } from "./ResultsView";
import { Banner, Card, EmptyState } from "./ui";

export function ReplayList() {
  const [runs, setRuns] = useState<ReplayInfo[] | null>(null);
  useEffect(() => {
    api.replays().then(setRuns).catch(() => setRuns([]));
  }, []);
  if (runs === null) return null;
  if (runs.length === 0)
    return (
      <EmptyState title="No recorded runs yet">
        Record one with <code className="font-mono">python -m scripts.record_replay SESSION_ID NAME</code> after a good run.
      </EmptyState>
    );
  return (
    <ul className="grid gap-3 sm:grid-cols-2">
      {runs.map((r) => (
        <li key={r.name}>
          <Link href={`/replay/${r.name}`} className="card block h-full p-4 hover:border-accent">
            <p className="font-semibold">{r.file_name ?? r.name}</p>
            {r.problem_statement && <p className="mt-1 text-sm">“{r.problem_statement}”</p>}
            {r.score != null && r.baseline != null && (
              <p className="mt-2 text-xs text-muted">
                {r.task?.replace("_", " ")} · held-out {r.metric} {fmt(r.score, 3)} vs baseline {fmt(r.baseline, 3)} ·{" "}
                {r.beyond_chance === false ? "no signal beyond chance" : "real gain"}
              </p>
            )}
            <p className="mt-2 text-sm text-accent">Watch the recorded run →</p>
          </Link>
        </li>
      ))}
    </ul>
  );
}

export function ReplayView({ name }: { name: string }) {
  const [speed, setSpeed] = useState(4);
  const [started, setStarted] = useState(false);
  const [results, setResults] = useState<Results | null>(null);
  const events = useRunStore((s) => s.events);
  const onEvent = useCallback(
    (e: RunEvent) => {
      if (e.type === "completed") api.replayResults(name).then(setResults);
    },
    [name],
  );
  useEventStream(started ? `replay:${name}:${speed}` : null, { url: api.replayEventsUrl(name, speed), onEvent, enabled: started });
  return (
    <div className="space-y-4">
      <Banner tone="info" title={`REPLAY: ${name}`}>
        {DEMO
          ? "A real run, recorded with the full system and played back in your browser. Nothing is computed on this site."
          : "This is a recorded run streamed again, not a live analysis."}
      </Banner>
      {!started && (
        <Card title="Play the recording">
          <div className="flex items-center gap-3">
            <label className="text-sm">
              Speed{" "}
              <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="rounded border border-border bg-surface px-2 py-1">
                {[1, 2, 4, 10].map((s) => (
                  <option key={s} value={s}>
                    {s}x
                  </option>
                ))}
              </select>
            </label>
            <button className="btn btn-primary" onClick={() => setStarted(true)}>
              Play
            </button>
            <button className="btn btn-ghost" onClick={() => api.replayResults(name).then(setResults)}>
              Skip to results
            </button>
          </div>
        </Card>
      )}
      {started && !results && (
        <Card title="Live run (replay)">
          <EventFeed events={events} showTools={false} />
        </Card>
      )}
      {results && <ResultsView results={results} sessionId={null} />}
    </div>
  );
}
