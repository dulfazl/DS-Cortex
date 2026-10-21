export type Status =
  | "created"
  | "queued"
  | "running"
  | "awaiting_approval"
  | "awaiting_clarification"
  | "blocked"
  | "completed"
  | "failed"
  | "interrupted";

export interface Evidence {
  tool: string;
  inputs: Record<string, unknown>;
  result: Record<string, unknown>;
  run_id: string;
}

export interface Check {
  run_id: string;
  path: string;
  op: string;
  value: unknown;
}

export interface Claim {
  text: string;
  kind: string;
  evidence: Evidence[];
  checks: Check[];
  source: "rule" | "llm";
  verified: boolean | null;
  verify_note: string | null;
}

export interface Recommendation {
  agent: string;
  decision_point: string;
  recommendation: string;
  rationale: string;
  evidence: Evidence[];
  confidence: number;
  confidence_method: string;
  verified: boolean;
  claims: Claim[];
}

export interface LedgerEntry {
  step: number;
  decision_point: string;
  options_considered: string[];
  chosen: string;
  recommendation: Recommendation;
  decided_by: "agent_rule" | "supervisor" | "client" | "system";
  client_choice: string | null;
  overridden: boolean;
  timestamp: string;
  note: string | null;
}

export interface ColumnInfo {
  name: string;
  kind: string;
  dtype: string;
  missing_pct: number;
  n_unique: number;
  unique_ratio: number;
  stored_as_text: string | null;
  examples: string[];
}

export interface Profile {
  file: string;
  rows: number;
  columns: number;
  sampled: boolean;
  column_info: ColumnInfo[];
  candidate_targets: string[];
  datetime_columns: string[];
  missing_pct_total: number;
}

export interface TaskPlan {
  task_family: string;
  target: string | null;
  metric: string;
  secondary_metrics: string[];
  rationale: string;
  confidence: number;
  explainability_required: boolean;
  imbalanced: boolean;
}

export interface Pending {
  kind: "approval" | "clarification" | "blocker";
  key: string;
  payload: Record<string, unknown> & {
    question?: string;
    options?: string[];
    reason?: string;
    bundle?: string;
    blockers?: string[];
  };
}

export interface SessionSummary {
  id: string;
  status: Status;
  mode: "autopilot" | "review";
  file_name: string;
  problem_statement: string;
  created_at: string;
  profile: Profile | null;
  task_plan: TaskPlan | null;
  pending: Pending | null;
  current_stage: string | null;
  completed_stages: string[];
  chosen_id: string | null;
  error: string | null;
  running: boolean;
  queue_position: number | null;
  memory: { suggestions?: MemorySuggestion[]; nearest_distance?: number | null };
  usage: Record<string, number | string>;
}

export interface MemorySuggestion {
  model: string;
  distance: number;
  created_at: string;
  test_score: number;
  baseline_score: number;
  metric: string;
}

export interface RunEvent {
  type: string;
  seq: number;
  ts?: string;
  replay?: boolean;
  [k: string]: unknown;
}

export interface Objection {
  check: string;
  severity: string;
  message: string;
  claim: Claim;
  resolved: boolean;
}

export interface Candidate {
  id: string;
  name: string;
  model: string;
  spec: Record<string, unknown>;
  steps: string[];
  cv: { primary: string; primary_mean: number; primary_std: number; per_fold: number[]; fit_time: number } | null;
  origin: string;
  parent: string | null;
  interpretable: boolean;
  objections: Objection[];
  status: string;
  rationale: string;
}

export interface DebateTurn {
  round: number;
  agent: string;
  kind: string;
  text: string;
  claims: Claim[];
  data: Record<string, unknown>;
}

export interface Plot {
  id: string;
  title: string;
  group: string;
  relevance: number;
  caption: string;
  figure: { data: unknown[]; layout: Record<string, unknown> };
}

export interface Results {
  session: SessionSummary;
  plan: TaskPlan;
  audit: {
    summary: string;
    blockers: string[];
    findings: {
      column: string | null;
      issue_type: string;
      severity: string;
      recommended_action: string;
      rationale: string;
      confidence: number;
      confidence_method: string;
      evidence: Evidence[];
    }[];
    cleaning_plan: { action: string; column: string | null; stage: string; enabled: boolean; reason: string }[];
  };
  stats: { flags: { issue: string; detail: string; severity: string }[]; guidance: string[] };
  candidates: Candidate[];
  debate: {
    chosen: string;
    runner_up: string | null;
    rule: string;
    forced_tiebreak: boolean;
    transcript: DebateTurn[];
    verifier: Record<string, unknown>;
    notes: string[];
    recommendation: Recommendation;
  } | null;
  final: {
    chosen: string;
    runner_up: string | null;
    note: string;
    confidence: { confidence: number; stability: number; separation: number; effect: number; formula: string };
  };
  test: {
    metric: string;
    chosen: string;
    score: number;
    test: Record<string, number>;
    n_train: number;
    n_test: number;
    baseline: { score: number };
    runner_up?: { id: string; score: number };
    shadow?: { recommended: string; recommended_score: number; client: string; client_score: number };
  };
  calibration: { confidence: number; outcome: boolean; beat_baseline: boolean } | null;
  override: { recommended: string; client: string; warning: string | null } | null;
  explain: {
    importance?: { method: string; importance: Record<string, number>; substituted?: boolean; substitution_reason?: string };
    narrative?: { summary: string; items: { step: number; decision_point: string; text: string }[]; source: string };
    narrative_beginner?: { summary: string; items: { step: number; decision_point: string; text: string }[] };
    business?: {
      headline: string;
      kpis: { name: string; value: number; baseline: number; lift: string | null }[];
      recommendations: { text: string; cites: string }[];
      caveats: string[];
    };
    research?: { note: string; novelty: string; disclaimer: string };
  };
  plots: Plot[];
  ledger: LedgerEntry[];
  memory: { suggestions?: MemorySuggestion[]; outcome?: Record<string, unknown> };
  export: { files: string[] };
  clean: { rows: number; n_train: number; n_test: number; columns: string[] } | null;
  usage: Record<string, number | string>;
  replay?: boolean;
}

export interface Meta {
  llm: string;
  llm_available: boolean;
  max_upload_mb: number;
  resolution_table: { rule: string; conflict: string; default_resolution: string; override_condition: string }[];
  drift_limits: string;
  privacy: string;
}

export interface DriftResult {
  features: Record<string, { type?: string; test?: string; statistic?: number; p_value?: number; effect?: number; missing_shift?: number; drifted: boolean; detail?: string }>;
  drifted: string[];
  drift_score: number;
  alert: boolean;
  threshold: number;
  limits: string;
}
