"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";
import type { DriftResult } from "@/lib/types";
import { Banner, Card } from "./ui";

export function DriftPanel({ id }: { id: string }) {
  const [res, setRes] = useState<DriftResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const upload = async (f: File | undefined) => {
    if (!f) return;
    setBusy(true);
    setError(null);
    try {
      setRes(await api.drift(id, f));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="space-y-4">
      <Card title="Drift check on a new batch">
        <p className="text-sm text-muted">
          Upload a new CSV with the same columns. Each feature is compared with the training data (KS test for continuous features, chi-square for
          categorical ones, and the change in missing values).
        </p>
        <p className="mt-1 text-xs text-muted">
          Limits: a two-sample test only. It does not retrain the model, monitor live traffic or track model performance.
        </p>
        <label className="btn btn-ghost mt-3 cursor-pointer">
          {busy ? "Checking…" : "Choose a batch CSV"}
          <input type="file" accept=".csv" className="hidden" onChange={(e) => upload(e.target.files?.[0])} disabled={busy} />
        </label>
        {error && <p className="mt-2 text-sm text-bad">{error}</p>}
      </Card>
      {res && (
        <Card title={`Drift score ${fmt(res.drift_score, 2)} (alert at ${res.threshold})`}>
          {res.alert ? (
            <Banner tone="bad" title="Drift alert">
              {res.drifted.length} feature(s) shifted: {res.drifted.join(", ")}. Re-check the model before relying on it for this batch.
            </Banner>
          ) : (
            <Banner tone="good" title="No drift alert">
              {res.drifted.length ? `Shifted: ${res.drifted.join(", ")}, below the alert threshold.` : "No feature shifted meaningfully."}
            </Banner>
          )}
          <div className="mt-3 overflow-x-auto">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Test</th>
                  <th>Statistic</th>
                  <th>p-value</th>
                  <th>Effect</th>
                  <th>Missing shift</th>
                  <th>Drifted</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(res.features).map(([k, f]) => (
                  <tr key={k} className={f.drifted ? "bg-bad-soft" : undefined}>
                    <td className="font-medium">{k}</td>
                    <td>{f.test ?? f.detail}</td>
                    <td className="tabular-nums">{fmt(f.statistic, 3)}</td>
                    <td className="tabular-nums">{f.p_value !== undefined ? f.p_value.toExponential(2) : "–"}</td>
                    <td className="tabular-nums">{fmt(f.effect, 3)}</td>
                    <td className="tabular-nums">{fmt(f.missing_shift, 3)}</td>
                    <td>{f.drifted ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
