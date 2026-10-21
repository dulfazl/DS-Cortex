import type { DriftResult, LedgerEntry, Meta, Profile, Results, SessionSummary } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
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
  replays: () => request<string[]>("/replays"),
  replayResults: (name: string) => request<Results>(`/replays/${name}/results`),
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
