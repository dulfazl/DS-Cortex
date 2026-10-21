"use client";

import type { Profile, TaskPlan } from "@/lib/types";
import { Card } from "./ui";

const KIND_STYLE: Record<string, string> = {
  numeric: "bg-accent-soft text-accent",
  categorical: "bg-good-soft text-good",
  boolean: "bg-good-soft text-good",
  datetime: "bg-warn-soft text-warn",
  id_like: "bg-bad-soft text-bad",
  constant: "bg-bad-soft text-bad",
  empty: "bg-bad-soft text-bad",
  text: "bg-surface-2 text-muted",
};

export function QuickProfile({ profile, plan }: { profile: Profile; plan: TaskPlan | null }) {
  const stats = [
    ["Rows", profile.rows.toLocaleString() + (profile.sampled ? " (sampled)" : "")],
    ["Columns", String(profile.columns)],
    ["Missing cells", `${profile.missing_pct_total.toFixed(1)}%`],
    ["Guessed target", plan?.target ?? profile.candidate_targets[0] ?? "none"],
    ["Task", plan ? `${plan.task_family.replace("_", " ")} · ${plan.metric}` : "decided when the run starts"],
  ];
  return (
    <Card title="Quick profile">
      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        {stats.map(([k, v]) => (
          <div key={k} className="rounded-lg bg-surface-2 px-3 py-2">
            <dt className="text-xs text-muted">{k}</dt>
            <dd className="truncate text-sm font-semibold" title={v}>
              {v}
            </dd>
          </div>
        ))}
      </dl>
      <div className="mt-4 max-h-80 overflow-auto">
        <table className="tbl">
          <thead>
            <tr>
              <th>Column</th>
              <th>Type</th>
              <th>Missing</th>
              <th>Distinct</th>
              <th>Examples</th>
            </tr>
          </thead>
          <tbody>
            {profile.column_info.map((c) => (
              <tr key={c.name}>
                <td className="font-medium">
                  {c.name}
                  {profile.candidate_targets.includes(c.name) && <span className="ml-1 text-xs text-accent">possible target</span>}
                </td>
                <td>
                  <span className={`rounded px-1.5 py-0.5 text-[11px] font-semibold ${KIND_STYLE[c.kind] ?? ""}`}>{c.kind.replace("_", " ")}</span>
                  {c.stored_as_text && <span className="ml-1 text-[11px] text-warn">{c.stored_as_text} as text</span>}
                </td>
                <td className={c.missing_pct > 20 ? "text-warn" : ""}>{c.missing_pct.toFixed(1)}%</td>
                <td>{c.n_unique.toLocaleString()}</td>
                <td className="max-w-xs truncate text-xs text-muted">{c.examples.join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
