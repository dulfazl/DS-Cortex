"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Meta } from "@/lib/types";

export function MetaBadge() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [down, setDown] = useState(false);
  useEffect(() => {
    // a sleeping free-plan API refuses or stalls for up to a minute: retry before calling it unreachable
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = (attempt: number) =>
      api
        .meta()
        .then((m) => !cancelled && setMeta(m))
        .catch(() => {
          if (cancelled) return;
          if (attempt >= 12) setDown(true);
          else timer = setTimeout(() => load(attempt + 1), 10_000);
        });
    load(0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);
  if (down) return <span className="rounded-full bg-bad-soft px-2.5 py-0.5 text-xs font-semibold text-bad">API unreachable</span>;
  if (!meta) return null;
  return (
    <span
      className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${meta.llm_available ? "bg-accent-soft text-accent" : "bg-surface-2 text-muted"}`}
      title={meta.llm_available ? "Gemini plans, critiques and narrates; tools compute every number" : "No Gemini key: deterministic templates narrate"}
    >
      {meta.llm_available ? meta.llm.replace("gemini:", "Gemini · ") : "Offline mode (no LLM)"}
    </span>
  );
}
