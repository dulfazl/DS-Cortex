import { create } from "zustand";
import type { RunEvent } from "./types";

interface RunState {
  sessionId: string | null;
  events: RunEvent[];
  lastSeq: number;
  connected: boolean;
  reset: (sessionId: string) => void;
  add: (ev: RunEvent) => void;
  setConnected: (c: boolean) => void;
}

/** One live run at a time: events appended in sequence order, duplicates dropped. */
export const useRunStore = create<RunState>((set, get) => ({
  sessionId: null,
  events: [],
  lastSeq: 0,
  connected: false,
  reset: (sessionId) => {
    if (get().sessionId !== sessionId) set({ sessionId, events: [], lastSeq: 0, connected: false });
  },
  add: (ev) => {
    if (ev.seq <= get().lastSeq) return;
    set((s) => ({ events: [...s.events, ev], lastSeq: ev.seq }));
  },
  setConnected: (connected) => set({ connected }),
}));
