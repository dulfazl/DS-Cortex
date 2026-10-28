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

    // "static:<path>?speed=N": a recorded run served as a file (demo site); play it back here at its pace
    const playStatic = (src: string) => {
      const [path, query] = src.slice("static:".length).split("?");
      const speed = Math.max(0.1, Number(new URLSearchParams(query).get("speed") || 1));
      fetch(path)
        .then((r) => r.json())
        .then((data: { events: { type: string; seq: number; offset: number; data: Record<string, unknown> }[] }) => {
          if (closed) return;
          setConnected(true);
          let last = 0;
          const play = (i: number) => {
            if (closed || i >= data.events.length) return;
            const e = data.events[i];
            const wait = Math.max(0, ((e.offset - last) / speed) * 1000);
            last = e.offset;
            retry = setTimeout(() => {
              const ev = { ...e.data, type: e.type, seq: e.seq, replay: true } as RunEvent;
              add(ev);
              onEvent.current?.(ev);
              if (TERMINAL.has(e.type)) {
                closed = true;
                setConnected(false);
                return;
              }
              play(i + 1);
            }, wait);
          };
          play(0);
        })
        .catch(() => setConnected(false));
    };

    const connect = () => {
      const after = useRunStore.getState().lastSeq;
      const base = opts?.url ?? `${API_URL}/sessions/${sessionId}/events`;
      if (base.startsWith("static:")) return playStatic(base);
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
