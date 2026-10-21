"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { StatusPill } from "./ui";

export function RecentSessions() {
  const [rows, setRows] = useState<{ id: string; status: string; created_at: string }[] | null>(null);
  useEffect(() => {
    api.listSessions().then((r) => setRows(r.slice(0, 8))).catch(() => setRows([]));
  }, []);
  if (!rows || rows.length === 0) return null;
  return (
    <section className="card p-4">
      <h2 className="text-sm font-semibold">Recent runs</h2>
      <ul className="mt-2 divide-y divide-border">
        {rows.map((r) => (
          <li key={r.id}>
            <Link href={`/sessions/${r.id}`} className="flex items-center gap-2 py-2 text-sm hover:text-accent">
              <span className="font-mono text-xs text-muted">{r.id.slice(0, 8)}</span>
              <span className="text-xs text-muted">{new Date(r.created_at).toLocaleString()}</span>
              <span className="ml-auto">
                <StatusPill status={r.status} />
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}
