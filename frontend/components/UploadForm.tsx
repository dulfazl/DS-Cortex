"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { Meta } from "@/lib/types";
import { Banner } from "./ui";

const EXAMPLES = [
  "Predict whether a customer will churn next month",
  "Estimate the sale price of each house",
  "Segment our customers into groups",
  "Forecast monthly sales for next quarter",
];

export function UploadForm() {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [statement, setStatement] = useState("");
  const [mode, setMode] = useState<"autopilot" | "review">("autopilot");
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);

  useEffect(() => {
    api.meta().then(setMeta).catch(() => undefined);
  }, []);

  const pick = (f: File | undefined | null) => {
    setError(null);
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".csv")) {
      setError("Only .csv files are accepted.");
      return;
    }
    if (meta && f.size > meta.max_upload_mb * 1024 * 1024) {
      setError(`The file is larger than ${meta.max_upload_mb} MB.`);
      return;
    }
    setFile(f);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !statement.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const { session_id } = await api.createSession(file, statement.trim(), mode);
      router.push(`/sessions/${session_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="card space-y-5 p-5">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          pick(e.dataTransfer.files?.[0]);
        }}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Choose a CSV file"
        className={`grid cursor-pointer place-items-center rounded-xl border-2 border-dashed px-4 py-10 text-center transition ${
          drag ? "border-accent bg-accent-soft" : "border-border hover:bg-surface-2"
        }`}
      >
        <input ref={input} type="file" accept=".csv,text/csv" className="hidden" onChange={(e) => pick(e.target.files?.[0])} data-testid="file-input" />
        {file ? (
          <div>
            <p className="font-semibold">{file.name}</p>
            <p className="text-sm text-muted">{(file.size / 1024).toFixed(1)} KB · click to change</p>
          </div>
        ) : (
          <div>
            <p className="font-semibold">Drop a CSV here, or click to choose</p>
            <p className="text-sm text-muted">Up to {meta?.max_upload_mb ?? 50} MB. The file is profiled, never executed.</p>
          </div>
        )}
      </div>

      <div>
        <label htmlFor="statement" className="text-sm font-semibold">
          What do you want to know?
        </label>
        <textarea
          id="statement"
          value={statement}
          onChange={(e) => setStatement(e.target.value)}
          rows={3}
          maxLength={2000}
          placeholder="e.g. Predict whether a customer will churn, and explain the main drivers"
          className="mt-1 w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
        />
        <div className="mt-2 flex flex-wrap gap-2">
          {EXAMPLES.map((x) => (
            <button key={x} type="button" onClick={() => setStatement(x)} className="rounded-full bg-surface-2 px-2.5 py-1 text-xs text-muted hover:text-text">
              {x}
            </button>
          ))}
        </div>
      </div>

      <fieldset>
        <legend className="text-sm font-semibold">Run mode</legend>
        <div className="mt-2 grid gap-2 sm:grid-cols-2">
          {(
            [
              ["autopilot", "Autopilot", "Runs end to end; stops only for blockers such as leakage or an unclear task."],
              ["review", "Review", "Pauses at preprocessing, features, modelling and tuning for your approval."],
            ] as const
          ).map(([value, title, text]) => (
            <label
              key={value}
              className={`cursor-pointer rounded-lg border p-3 ${mode === value ? "border-accent bg-accent-soft" : "border-border"}`}
            >
              <input type="radio" name="mode" value={value} checked={mode === value} onChange={() => setMode(value)} className="sr-only" />
              <p className="text-sm font-semibold">{title}</p>
              <p className="text-xs text-muted">{text}</p>
            </label>
          ))}
        </div>
      </fieldset>

      {meta?.llm_available && <Banner tone="warn" title="Privacy">{meta.privacy}</Banner>}
      {error && <Banner tone="bad" title="Upload failed">{error}</Banner>}

      <div className="flex items-center gap-3">
        <button type="submit" className="btn btn-primary" disabled={!file || !statement.trim() || busy}>
          {busy ? "Uploading and profiling…" : "Upload and profile"}
        </button>
        <p className="text-xs text-muted">You will see the quick profile before anything runs.</p>
      </div>
    </form>
  );
}
