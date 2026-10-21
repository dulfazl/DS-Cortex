"use client";

import { ConfidenceBadge } from "./ConfidenceBadge";
import type { EvidencePayload } from "./EvidenceDrawer";
import { AGENT_NAMES, label } from "@/lib/format";
import type { LedgerEntry } from "@/lib/types";

const WHO: Record<string, string> = {
  agent_rule: "agent rule",
  supervisor: "supervisor",
  client: "you",
  system: "system",
};

export function LedgerTable({ entries, onEvidence }: { entries: LedgerEntry[]; onEvidence: (p: EvidencePayload) => void }) {
  return (
    <div className="overflow-x-auto">
      <table className="tbl">
        <thead>
          <tr>
            <th>Step</th>
            <th>Decision point</th>
            <th>Chosen</th>
            <th>Options considered</th>
            <th>Confidence</th>
            <th>Decided by</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => {
            const r = e.recommendation;
            return (
              <tr key={e.step} className={e.overridden ? "bg-warn-soft" : undefined}>
                <td className="font-mono text-muted">{e.step}</td>
                <td>
                  <div className="font-medium">{label(e.decision_point)}</div>
                  <div className="text-xs text-muted">{AGENT_NAMES[r.agent] ?? r.agent}</div>
                </td>
                <td className="max-w-xs">
                  <div className="font-medium">{e.chosen}</div>
                  <div className="mt-0.5 line-clamp-3 text-xs text-muted" title={r.rationale}>
                    {r.rationale}
                  </div>
                  {e.note && <div className="mt-0.5 text-xs text-warn">{e.note}</div>}
                </td>
                <td className="max-w-[220px] text-xs text-muted">{e.options_considered.slice(0, 6).join(", ")}</td>
                <td>
                  <ConfidenceBadge
                    value={r.confidence}
                    method={r.confidence_method}
                    onClick={() =>
                      onEvidence({
                        title: `Step ${e.step}: ${label(e.decision_point)}`,
                        subtitle: r.rationale,
                        evidence: r.evidence,
                        claims: r.claims,
                      })
                    }
                  />
                </td>
                <td className="text-xs">
                  {WHO[e.decided_by] ?? e.decided_by}
                  {e.overridden && <div className="font-semibold text-warn">overridden</div>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
