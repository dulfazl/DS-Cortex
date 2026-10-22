"use client";

import { useEffect, useRef } from "react";
import { API_URL, EVENT_TYPES } from "./api";
import { useRunStore } from "./store";
import type { RunEvent } from "./types";

const TERMINAL = new Set(["completed", "failed"]);

/**
 * Subscribes to the run's Server-Sent Events. Every event carries a sequence number; on a dropped
 * connection we reconnect with the last one we saw, so nothing is lost or repeated.
 * `url` lets the replay page stream a recorded run through the same code. `onDrop` fires when the
 * connection fails and again when it comes back, so the page can re-read the run: a server restart
 * changes its status (or, without a disk, removes it) without sending an event.
 */
export function useEventStream(
  sessionId: string | null,
  opts?: { url?: string; enabled?: boolean; onEvent?: (e: RunEvent) => void; onDrop?: () => void },
) {
  const add = useRunStore((s) => s.add);
  const reset = useRunStore((s) => s.reset);
  const setConnected = useRunStore((s) => s.setConnected);
  const onEvent = useRef(opts?.onEvent);
  const onDrop = useRef(opts?.onDrop);
  useEffect(() => {
    onEvent.current = opts?.onEvent;
    onDrop.current = opts?.onDrop;
  });
  const enabled = opts?.enabled ?? true;

  useEffect(() => {
    if (!sessionId || !enabled) return;
    reset(sessionId);
    let es: EventSource | null = null;
    let closed = false;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let dropped = false;

    const connect = () => {
      const after = useRunStore.getState().lastSeq;
      const base = opts?.url ?? `${API_URL}/sessions/${sessionId}/events`;
      es = new EventSource(`${base}${base.includes("?") ? "&" : "?"}after=${after}`);
      es.onopen = () => {
        setConnected(true);
        if (dropped) onDrop.current?.();
        dropped = false;
      };
      const handler = (type: string) => (msg: MessageEvent) => {
        const data = JSON.parse(msg.data);
        const ev: RunEvent = { ...data, type, seq: Number(msg.lastEventId || data.seq) };
        add(ev);
        onEvent.current?.(ev);
        if (TERMINAL.has(type)) {
          closed = true;
          es?.close();
          setConnected(false);
        }
      };
      for (const t of EVENT_TYPES) es.addEventListener(t, handler(t));
      es.onerror = () => {
        es?.close();
        setConnected(false);
        dropped = true;
        if (closed) return;
        onDrop.current?.();
        retry = setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      es?.close();
      setConnected(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, enabled, opts?.url]);
}
