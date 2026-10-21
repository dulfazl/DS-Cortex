"""The one contract every agent follows. Built before any agent exists (Phase 1)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TaskFamily = Literal["classification", "regression", "clustering", "anomaly_detection", "time_series", "offline_rl"]
Severity = Literal["low", "medium", "high", "blocker"]
ConfidenceMethod = Literal["cv_variance", "bootstrap", "effect_size", "rule", "llm_estimate"]
DecidedBy = Literal["agent_rule", "supervisor", "client", "system"]


class EvidenceError(ValueError):
    """A recommendation broke the evidence rule."""


# --------------------------------------------------------------------------- evidence
class Evidence(BaseModel):
    tool: str                       # e.g. "stats_tools.vif_report"
    inputs: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any]          # the real executed output
    run_id: str


class Check(BaseModel):
    """One machine-checkable condition linking a claim to a logged tool result."""
    run_id: str
    path: str                       # dotted path into the logged result, e.g. "candidates.rf.primary_mean"
    op: Literal["approx", "eq", "gt", "ge", "lt", "le", "true", "false", "contains"]
    value: Any = None
    tol: float = 1e-3


class Claim(BaseModel):
    text: str
    evidence: list[Evidence] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    source: Literal["rule", "llm"] = "rule"
    verified: bool | None = None
    verify_note: str | None = None


class Recommendation(BaseModel):
    agent: str
    decision_point: str             # "imputation", "model_choice", ...
    recommendation: str
    rationale: str
    evidence: list[Evidence]        # empty list = claim is rejected
    confidence: float = Field(ge=0, le=1)
    confidence_method: ConfidenceMethod
    verified: bool = False
    claims: list[Claim] = Field(default_factory=list)


class LedgerEntry(BaseModel):
    step: int
    decision_point: str
    options_considered: list[str]
    chosen: str
    recommendation: Recommendation
    decided_by: DecidedBy = "agent_rule"
    client_choice: str | None = None
    overridden: bool = False
    timestamp: str
    note: str | None = None


def validate_recommendation(rec: Recommendation, computed_available: bool = True) -> Recommendation:
    """Every agent output passes through here. No evidence, or an LLM-stated confidence where a
    computed method was available, is rejected."""
    if not rec.evidence:
        raise EvidenceError(f"{rec.agent}/{rec.decision_point}: recommendation has no evidence")
    for ev in rec.evidence:
        if not ev.run_id or ev.result is None:
            raise EvidenceError(f"{rec.agent}/{rec.decision_point}: evidence without a tool run")
    if rec.confidence_method == "llm_estimate" and computed_available:
        raise EvidenceError(f"{rec.agent}/{rec.decision_point}: confidence must be computed, not stated by the LLM")
    return rec


# --------------------------------------------------------------------------- router
class TaskPlan(BaseModel):
    task_family: TaskFamily
    target: str | None = None
    metric: str
    secondary_metrics: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0, le=1)
    confidence_method: ConfidenceMethod = "rule"
    imbalanced: bool = False
    time_column: str | None = None
    group_column: str | None = None
    action_column: str | None = None
    reward_column: str | None = None
    propensity_column: str | None = None
    explainability_required: bool = False
    notes: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class ClarifyingQuestion(BaseModel):
    question: str
    options: list[str]
    reason: str
    kind: Literal["target", "task_family", "other"] = "target"


# --------------------------------------------------------------------------- auditor
class AuditFinding(BaseModel):
    column: str | None
    issue_type: str
    severity: Severity
    recommended_action: str
    rationale: str
    confidence: float = Field(ge=0, le=1)
    confidence_method: ConfidenceMethod
    evidence: list[Evidence]


class CleaningAction(BaseModel):
    action: str                     # drop_column, drop_duplicates, recode_case, strip_whitespace, ...
    column: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    stage: Literal["cleaning", "pipeline"] = "cleaning"   # pipeline actions are fitted inside CV
    enabled: bool = True


class DataAuditReport(BaseModel):
    findings: list[AuditFinding]
    cleaning_plan: list[CleaningAction]
    blockers: list[str] = Field(default_factory=list)
    summary: str = ""


# --------------------------------------------------------------------------- statistician
class FeatureProfile(BaseModel):
    name: str
    kind: str
    mean: float | None = None
    std: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    normality_test: str | None = None
    normal_p: float | None = None
    best_fit: str | None = None
    best_fit_ks: float | None = None
    effect_measure: str | None = None
    effect_size: float | None = None
    p_value: float | None = None


class StatFlag(BaseModel):
    issue: str
    features: list[str]
    detail: str
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    confidence_method: ConfidenceMethod
    evidence: list[Evidence]


class StatisticalProfile(BaseModel):
    features: list[FeatureProfile]
    flags: list[StatFlag]
    guidance: list[str]
    signals: dict[str, Any]
    top_correlations: list[dict[str, Any]] = Field(default_factory=list)
    fingerprint: list[float] = Field(default_factory=list)
    fingerprint_features: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- modelling
class CVScores(BaseModel):
    primary: str
    primary_mean: float
    primary_std: float
    per_fold: list[float]
    secondary: dict[str, float] = Field(default_factory=dict)
    train_mean: float | None = None
    fit_time: float = 0.0
    n_splits: int = 0
    baseline_mean: float | None = None


class Objection(BaseModel):
    check: str
    severity: Severity
    message: str
    claim: Claim
    resolved: bool = False


class CandidatePipeline(BaseModel):
    id: str
    name: str
    model: str
    spec: dict[str, Any]
    steps: list[str] = Field(default_factory=list)
    cv: CVScores | None = None
    origin: Literal["factory", "memory", "revision", "tuned"] = "factory"
    parent: str | None = None
    interpretable: bool = False
    objections: list[Objection] = Field(default_factory=list)
    status: Literal["active", "rejected", "finalist", "chosen"] = "active"
    rationale: str = ""
    eval_run_id: str | None = None


class DebatePosition(BaseModel):
    agent: str
    preferred: str
    reasons: list[Claim]
    confidence: float = Field(ge=0, le=1)
    confidence_method: ConfidenceMethod
    round: int = 0
    excluded: bool = False


class DebateTurn(BaseModel):
    round: int
    agent: str
    kind: Literal["position", "verification", "conflict", "resolution", "rebuttal", "decision", "note"]
    text: str
    claims: list[Claim] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


class DebateOutcome(BaseModel):
    chosen: str
    runner_up: str | None
    rule: str
    forced_tiebreak: bool = False
    positions: list[DebatePosition]
    transcript: list[DebateTurn]
    verifier: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    recommendation: Recommendation
