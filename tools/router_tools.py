"""Rule-based pre-checks that run before the LLM and give it facts to reason over.
The task-to-metric table is fixed here, before anything runs, so a metric cannot be picked
later because it flatters a result."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

METRICS: dict[str, dict[str, Any]] = {
    "classification": {"primary": "f1_macro", "secondary": ["roc_auc"]},
    "classification_imbalanced": {"primary": "pr_auc", "secondary": ["recall_at_precision"]},
    "regression": {"primary": "rmse", "secondary": ["r2", "mae"]},
    "clustering": {"primary": "silhouette", "secondary": ["davies_bouldin"]},
    "anomaly_detection": {"primary": "pr_auc", "secondary": []},
    "anomaly_detection_unlabeled": {"primary": "score_stability", "secondary": []},
    "time_series": {"primary": "smape", "secondary": ["mae"]},
    "offline_rl": {"primary": "policy_value", "secondary": []},
}
LOWER_IS_BETTER = {"rmse", "mae", "smape", "davies_bouldin"}

KEYWORDS = {
    "forecast": ["forecast", "next month", "next week", "next quarter", "next year", "next day", "future",
                 "over time", "time series", "upcoming", "projection", "project sales", "predict demand"],
    "cluster": ["segment", "segments", "segmentation", "cluster", "clusters", "clustering", "group similar",
                "groups of", "personas", "persona", "natural groups"],
    "anomaly": ["unusual", "anomal", "fraud", "outlier", "suspicious", "abnormal", "odd ", "rare event", "defect"],
    "rl": ["policy", "best action", "which action", "which offer", "reinforcement", "bandit", "maximize reward",
           "maximise reward", "treatment to give", "next best action", "which treatment", "which arm"],
    "explain": ["explain", "interpretable", "interpretability", "why ", "drivers", "transparent", "explainable",
                "understand what", "reasons"],
    "predict": ["predict", "forecast", "estimate", "classify", "whether", "determine", "model ", "detect", "flag",
                "will ", "which customers", "likelihood", "probability"],
}
ACTION_NAMES = ("action", "treatment", "arm", "offer", "decision", "intervention", "campaign")
REWARD_NAMES = ("reward", "payoff", "return", "gain", "outcome", "profit", "clicked", "converted", "response")
PROPENSITY_NAMES = ("propensity", "prob", "probability", "logging_prob", "p_action", "pscore")
UNSUITABLE_TARGET_KINDS = {"id_like", "constant", "empty", "text", "datetime"}


def _norm(s: str) -> str:
    return re.sub(r"[_\-]+", " ", s.lower()).strip()


def mentioned_columns(statement: str, columns: list[str]) -> list[tuple[int, str]]:
    st = " " + _norm(statement) + " "
    found = []
    for c in columns:
        variants = {_norm(c), c.lower()}
        for v in variants:
            if not v:
                continue
            m = re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", st)
            if m:
                found.append((m.start(), c))
                break
    return sorted(found)


def keyword_hits(statement: str) -> dict[str, list[str]]:
    st = " " + statement.lower() + " "
    return {k: [w for w in words if w in st] for k, words in KEYWORDS.items()}


def _named_like(col: str, names: tuple[str, ...]) -> bool:
    n = col.lower()
    return any(n == x or n.startswith(x + "_") or n.endswith("_" + x) or n.startswith(x) for x in names)


def rl_columns(columns: list[str]) -> dict[str, str | None]:
    action = next((c for c in columns if _named_like(c, ACTION_NAMES)), None)
    reward = next((c for c in columns if _named_like(c, REWARD_NAMES) and c != action), None)
    prop = next((c for c in columns if _named_like(c, PROPENSITY_NAMES)), None)
    return {"action": action, "reward": reward, "propensity": prop}


def target_type(df: pd.DataFrame, target: str, kind: str) -> dict:
    y = df[target].dropna()
    nunique = int(y.nunique())
    numeric = pd.api.types.is_numeric_dtype(y)
    if kind == "boolean" or nunique == 2:
        ttype = "binary"
    elif not numeric or (nunique <= 20 and (y % 1 == 0).all() and nunique <= max(10, int(0.05 * len(y)))):
        ttype = "multiclass" if nunique <= 50 else "high_cardinality"
    else:
        ttype = "continuous"
    out: dict[str, Any] = {"target": target, "type": ttype, "n_unique": nunique}
    if ttype in ("binary", "multiclass"):
        shares = y.value_counts(normalize=True)
        out["class_shares"] = {str(k): round(float(v), 4) for k, v in shares.items()}
        out["minority_share"] = round(float(shares.min()), 4)
        out["imbalance_ratio"] = round(float(shares.max() / shares.min()), 3)
    return out


def precheck(statement: str, profile: dict, df: pd.DataFrame) -> dict:
    """Facts for the Router: mentions, keyword hits, RL columns, target type, suggested family."""
    cols = [c["name"] for c in profile["column_info"]]
    kinds = {c["name"]: c["kind"] for c in profile["column_info"]}
    mentions = mentioned_columns(statement, cols)
    hits = keyword_hits(statement)
    st = statement.lower()

    # target: explicit pattern, else first suitable mention after a predictive verb, else first suitable mention
    target, target_source = None, None
    m = re.search(r"(?:target(?: column)?(?: is|:|=)?|label(?: is|:)?)\s+[`'\"]?([\w\- ]+?)[`'\"]?(?:[\s,.;]|$)", st)
    if m:
        cand = next((c for c in cols if _norm(c) == _norm(m.group(1))), None)
        if cand:
            target, target_source = cand, "explicit"
    suitable = [(p, c) for p, c in mentions if kinds.get(c) not in UNSUITABLE_TARGET_KINDS]
    if target is None and suitable:
        verb_pos = min((st.find(w) for w in KEYWORDS["predict"] if w in st), default=-1)
        after = [c for p, c in suitable if verb_pos >= 0 and p >= verb_pos]
        target = after[0] if after else suitable[0][1]
        target_source = "mentioned"
    hint_targets = [c for c in profile.get("candidate_targets", []) if kinds.get(c) not in UNSUITABLE_TARGET_KINDS]

    rl = rl_columns(cols)
    rl_data = bool(rl["action"] and rl["reward"])
    dt_cols = profile.get("datetime_columns", [])
    tinfo = target_type(df, target, kinds[target]) if target else None

    family, conf, reasons = None, 0.0, []
    if hits["rl"] or (rl_data and not target):
        if rl_data:
            family, conf = "offline_rl", 0.9 if hits["rl"] else 0.75
            reasons.append(f"logged decisions found: action column '{rl['action']}', reward column '{rl['reward']}'")
            target = rl["reward"]
            target_source = target_source or "rl_reward"
        else:
            reasons.append("RL wording but no action/reward columns: the data does not look like logged decisions")
    if family is None and hits["forecast"] and dt_cols and tinfo and tinfo["type"] == "continuous":
        family, conf = "time_series", 0.9
        reasons.append(f"forecast wording, datetime column '{dt_cols[0]}' and a numeric target")
    if family is None and hits["anomaly"]:
        if tinfo and tinfo["type"] == "binary":
            if tinfo["minority_share"] < 0.05:
                family, conf = "anomaly_detection", 0.85
                reasons.append(f"anomaly wording with a rare binary label ({tinfo['minority_share']:.1%})")
            else:
                family, conf = "classification", 0.85
                reasons.append("anomaly wording but the labels are not rare: treat as classification")
        elif not target:
            family, conf = "anomaly_detection", 0.85
            reasons.append("anomaly wording and no label column named")
    if family is None and hits["cluster"] and not target:
        family, conf = "clustering", 0.88
        reasons.append("segmentation wording and no target named")
    if family is None and tinfo:
        if tinfo["type"] in ("binary", "multiclass"):
            family = "classification"
            reasons.append(f"target '{target}' is {tinfo['type']} with {tinfo['n_unique']} classes")
        elif tinfo["type"] == "continuous":
            family = "regression"
            reasons.append(f"target '{target}' is continuous numeric")
        else:
            reasons.append(f"target '{target}' has {tinfo['n_unique']} levels: too many classes")
        if family:
            conf = 0.9 if target_source in ("explicit", "mentioned") else 0.55
    if family is None and not target and not any(hits[k] for k in ("cluster", "anomaly", "forecast", "rl")):
        reasons.append("no target named and no task wording")

    ambiguous = []
    if target_source == "mentioned" and len([c for _, c in suitable]) > 1 and not hits["predict"]:
        ambiguous = [c for _, c in suitable]
    if family in ("time_series",) and len(dt_cols) > 1:
        reasons.append(f"several datetime columns; using '{dt_cols[0]}'")

    return {
        "mentioned_columns": [c for _, c in mentions],
        "keyword_hits": {k: v for k, v in hits.items() if v},
        "target": target,
        "target_source": target_source,
        "target_info": tinfo,
        "hint_targets": hint_targets,
        "rl_columns": rl,
        "rl_data": rl_data,
        "datetime_columns": dt_cols,
        "suggested_family": family,
        "rule_confidence": round(conf, 3),
        "ambiguous_targets": ambiguous,
        "explainability_requested": bool(hits["explain"]),
        "reasons": reasons,
    }


def metric_for(family: str, imbalanced: bool = False, labeled: bool = True) -> dict:
    if family == "classification" and imbalanced:
        return METRICS["classification_imbalanced"]
    if family == "anomaly_detection" and not labeled:
        return METRICS["anomaly_detection_unlabeled"]
    return METRICS[family]


def closest_supervised_framing(profile: dict, precheck_result: dict) -> str:
    t = precheck_result.get("target") or (precheck_result.get("hint_targets") or [None])[0]
    if not t:
        return "There is no outcome column, so clustering or anomaly detection is the closest framing."
    tinfo = precheck_result.get("target_info") or {}
    kind = "classification" if tinfo.get("type") in ("binary", "multiclass") else "regression"
    return f"The closest supervised framing is {kind} of '{t}' from the other columns."
