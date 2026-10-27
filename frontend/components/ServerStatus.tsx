"use client";

import { useEffect, useState } from "react";
import { API_URL } from "@/lib/api";
import { Banner } from "./ui";

type State = "ok" | "waking" | "down";

/**
 * The API runs on a free plan that sleeps after 15 idle minutes, and the first request then takes up to a
 * minute. Ping /healthz and say so while it wakes, instead of looking broken.
 */
export function ServerStatus() {
  const [state, setState] = useState<State>("ok");
  useEffect(() => {
    let cancelled = false;
    const slow = setTimeout(() => !cancelled && setState("waking"), 2500);
    const ping = async (attempt: number): Promise<void> => {
      try {
        const res = await fetch(`${API_URL}/healthz`, { cache: "no-store" });
        if (res.ok) {
          clearTimeout(slow);
          if (!cancelled) setState("ok");
          return;
        }
      } catch {
        /* asleep or unreachable: try again */
      }
      if (cancelled) return;
      if (attempt >= 12) {
        clearTimeout(slow);
        setState("down");   // about two minutes without an answer
        return;
      }
      setState("waking");
      await new Promise((r) => setTimeout(r, 10_000));
      return ping(attempt + 1);
    };
    ping(0);
    return () => {
      cancelled = true;
      clearTimeout(slow);
    };
  }, []);

  if (state === "ok") return null;
  return (
    <div className="mb-4">
      {state === "waking" ? (
        <Banner tone="info" title="Waking up the analysis server">
          It runs on a free plan that sleeps after 15 idle minutes, so the first request can take up to a minute. This page
          carries on by itself.
        </Banner>
      ) : (
        <Banner tone="bad" title="The analysis server is not responding">
          It may be down for maintenance or out of free hours this month. Try again later.
        </Banner>
      )}
    </div>
  );
}
