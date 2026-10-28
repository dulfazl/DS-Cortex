import type { DriftResult, LedgerEntry, Meta, Profile, Results, SessionSummary } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

/** Demo build (NEXT_PUBLIC_DEMO=1): no API server; recorded runs are static files under /demo. */
export const DEMO = process.env.NEXT_PUBLIC_DEMO === "1";

export type ReplayInfo = {
  name: string;
  file_name?: string;
  problem_statement?: string;
  task?: string | null;
  metric?: string | null;
  score?: number | null;
  baseline?: number | null;
  beyond_chance?: boolean | null;
  duration_seconds?: number;
};

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function staticJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new ApiError(res.status, `Could not load ${path}`);
  return res.json() as Promise<T>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (DEMO) throw new ApiError(0, "This is the demo site: it replays recorded runs and has no analysis server.");
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(0, `Cannot reach the DS Cortex API at ${API_URL}. Is the backend running?`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  meta: () => request<Meta>("/meta"),
  listSessions: () => request<{ id: string; status: string; created_at: string }[]>("/sessions"),
  createSession: (file: File, statement: string, mode: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("problem_statement", statement);
    fd.append("mode", mode);
    return request<{ session_id: string; profile: Profile }>("/sessions", { method: "POST", body: fd });
  },
  start: (id: string, mode?: string) => request<{ started: boolean; status: string }>(`/sessions/${id}/start`, json({ mode })),
  session: (id: string) => request<SessionSummary>(`/sessions/${id}`),
  clarify: (id: string, choice: string) => request<{ accepted: unknown }>(`/sessions/${id}/clarify`, json({ choice })),
  approve: (id: string, body: Record<string, unknown>) =>
    request<{ accepted: boolean; warning: string | null }>(`/sessions/${id}/approve`, json(body)),
  results: (id: string) => request<Results>(`/sessions/${id}/results`),
  ledger: (id: string) => request<LedgerEntry[]>(`/sessions/${id}/ledger`),
  chat: (id: string, question: string) =>
    request<{ answer: string; source: string; cited: { kind: string; step?: number; decision_point?: string }[] }>(
      `/sessions/${id}/chat`,
      json({ question }),
    ),
  drift: (id: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<DriftResult>(`/sessions/${id}/drift`, { method: "POST", body: fd });
  },
  rerun: (id: string) => request<{ session_id: string }>(`/sessions/${id}/rerun`, { method: "POST" }),
  exportUrl: (id: string, file?: string) => `${API_URL}/sessions/${id}/export${file ? `?file=${encodeURIComponent(file)}` : ""}`,
  replays: (): Promise<ReplayInfo[]> =>
    DEMO ? staticJson<ReplayInfo[]>("/demo/index.json") : request<string[]>("/replays").then((names) => names.map((name) => ({ name }))),
  replayResults: (name: string): Promise<Results> =>
    DEMO
      ? staticJson<{ results: Results }>(`/demo/${name}.json`).then((d) => ({ ...d.results, replay: true, replay_name: name }))
      : request<Results>(`/replays/${name}/results`),
  /** Event source for a replay: the API's stream, or the static recording played back in the browser. */
  replayEventsUrl: (name: string, speed: number) =>
    DEMO ? `static:/demo/${name}.json?speed=${speed}` : `${API_URL}/replays/${name}/events?speed=${speed}`,
};

export const EVENT_TYPES = [
  "agent_started",
  "agent_finished",
  "tool_called",
  "ledger_entry",
  "plot_ready",
  "approval_required",
  "clarification_required",
  "blocker",
  "completed",
  "failed",
  "status",
  "queued",
  "stage",
  "candidate_evaluated",
  "critic_review",
  "critic_round",
  "memory_suggestion",
  "warning",
] as const;
