"use client";

import { useState } from "react";
import { api } from "@/lib/api";

interface Msg {
  role: "you" | "cortex";
  text: string;
  cited?: string;
}

const SUGGESTED = ["Why was this model chosen?", "Why did you drop columns?", "Which metric is used and why?", "What did the critic object to?"];

/** Questions about THIS run only, answered from its ledger and report. */
export function ChatPanel({ id, onClose }: { id: string; onClose: () => void }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const ask = async (question: string) => {
    if (!question.trim()) return;
    setMsgs((m) => [...m, { role: "you", text: question }]);
    setQ("");
    setBusy(true);
    try {
      const a = await api.chat(id, question);
      const cited = a.cited
        .filter((c) => c.step)
        .map((c) => `step ${c.step}`)
        .join(", ");
      setMsgs((m) => [...m, { role: "cortex", text: a.answer, cited: cited ? `from ${cited}` : undefined }]);
    } catch (e) {
      setMsgs((m) => [...m, { role: "cortex", text: e instanceof Error ? e.message : String(e) }]);
    } finally {
      setBusy(false);
    }
  };
  return (
    <aside className="card flex h-[640px] flex-col p-0 lg:sticky lg:top-4" aria-label="Run chat">
      <div className="flex items-center border-b border-border px-4 py-3">
        <div>
          <p className="text-sm font-semibold">Ask about this run</p>
          <p className="text-xs text-muted">Answers come from this run&apos;s ledger and report only.</p>
        </div>
        <button className="ml-auto text-sm text-muted hover:text-text" onClick={onClose} aria-label="Close chat">
          ✕
        </button>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {msgs.length === 0 && (
          <div className="flex flex-wrap gap-2">
            {SUGGESTED.map((s) => (
              <button key={s} className="rounded-full bg-surface-2 px-2.5 py-1 text-xs text-muted hover:text-text" onClick={() => ask(s)}>
                {s}
              </button>
            ))}
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={m.role === "you" ? "text-right" : ""}>
            <div className={`inline-block max-w-[90%] rounded-xl px-3 py-2 text-left text-sm ${m.role === "you" ? "bg-accent text-white" : "bg-surface-2"}`}>
              {m.text}
              {m.cited && <p className="mt-1 text-[11px] opacity-70">{m.cited}</p>}
            </div>
          </div>
        ))}
        {busy && <p className="text-xs text-muted">Looking it up in the ledger…</p>}
      </div>
      <form
        className="flex gap-2 border-t border-border p-3"
        onSubmit={(e) => {
          e.preventDefault();
          ask(q);
        }}
      >
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Why did you drop this column?"
          className="min-w-0 flex-1 rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
          aria-label="Question"
        />
        <button className="btn btn-primary" disabled={busy || !q.trim()}>
          Ask
        </button>
      </form>
    </aside>
  );
}
