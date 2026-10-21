export const fmt = (v: unknown, digits = 3): string => {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "–";
    if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
    return v.toFixed(digits);
  }
  return String(v);
};

export const label = (s: string | null | undefined): string => (s ?? "").replace(/_/g, " ").replace(/:/g, " · ");

export const LOWER_IS_BETTER = new Set(["rmse", "mae", "smape", "davies_bouldin"]);

export const AGENT_NAMES: Record<string, string> = {
  router: "Router",
  data_auditor: "Data Auditor",
  statistician: "Statistician",
  ml_architect: "ML Architect",
  pipeline_critic: "Pipeline Critic",
  supervisor: "Supervisor",
  business: "Business Insight",
  research: "Research",
  explainability: "Explainability",
  report_generator: "Report",
  verifier: "Verifier",
};

export const STAGES: { key: string; label: string; agent: string }[] = [
  { key: "profile", label: "Quick profile", agent: "router" },
  { key: "route", label: "Task routing", agent: "router" },
  { key: "audit", label: "Data audit", agent: "data_auditor" },
  { key: "blockers", label: "Leakage check", agent: "data_auditor" },
  { key: "approve_preprocessing", label: "Preprocessing review", agent: "supervisor" },
  { key: "clean", label: "Cleaning and holdout", agent: "data_auditor" },
  { key: "stats", label: "Statistics and memory", agent: "statistician" },
  { key: "approve_features", label: "Feature review", agent: "supervisor" },
  { key: "candidates", label: "Candidate pipelines", agent: "ml_architect" },
  { key: "critic", label: "Critique and revision", agent: "pipeline_critic" },
  { key: "debate", label: "Debate", agent: "supervisor" },
  { key: "approve_modelling", label: "Model review", agent: "supervisor" },
  { key: "tune", label: "Tuning finalists", agent: "ml_architect" },
  { key: "final", label: "Held-out test", agent: "supervisor" },
  { key: "explain", label: "Explanations and plots", agent: "explainability" },
  { key: "report", label: "Report and export", agent: "report_generator" },
];
