"""Data Auditor: the quality gate every later agent trusts. It never fixes anything silently;
it proposes, justifies and logs. Confidence comes from the evidence, not from the LLM."""
from __future__ import annotations

import pandas as pd

from agents.base import BaseAgent, slice_evidence
from app.schemas import AuditFinding, CleaningAction, DataAuditReport, Evidence, TaskPlan
from tools import pandas_tools as pt
from tools import stats_tools as stt
from tools.common import clip01

SEVERITY_ORDER = {"blocker": 0, "high": 1, "medium": 2, "low": 3}
ACTION_ORDER = {"drop_duplicates": 0, "drop_rows_missing_target": 1, "drop_column": 2, "to_numeric": 3,
                "to_datetime": 3, "strip_whitespace": 4, "recode_case": 4, "set_nan_invalid": 5}


class DataAuditorAgent(BaseAgent):
    name = "data_auditor"

    def audit(self, df: pd.DataFrame, plan: TaskPlan) -> DataAuditReport:
        self.started("calling every audit tool")
        target, fam = plan.target, plan.task_family
        findings: list[AuditFinding] = []
        actions: list[CleaningAction] = []
        protected = {c for c in (target, plan.time_column, plan.action_column, plan.reward_column,
                                 plan.propensity_column, plan.group_column) if c}

        def add(col, issue, sev, action_text, why, conf, method, ev: Evidence, payload=None, action: CleaningAction | None = None):
            findings.append(AuditFinding(column=col, issue_type=issue, severity=sev, recommended_action=action_text,
                                         rationale=why, confidence=round(clip01(conf), 4), confidence_method=method,
                                         evidence=[slice_evidence(ev, col, payload) if payload is not None else ev]))
            if action:
                actions.append(action)

        # ---- types
        types, ev = self.ctx.tool("pandas_tools.type_inference", pt.type_inference, {"columns": list(df.columns)}, df)
        for col, info in types["columns"].items():
            issue = info["issue"]
            if not issue or col in protected:
                continue
            if issue == "id_like":
                add(col, "id_like", "high", "drop the column", f"{info['unique_ratio']:.0%} of values are unique: an identifier, not a feature",
                    info["unique_ratio"], "rule", ev, info, CleaningAction(action="drop_column", column=col, reason="identifier"))
            elif issue in ("constant", "empty"):
                add(col, issue, "medium", "drop the column", f"the column is {issue}: it carries no information",
                    1.0, "rule", ev, info, CleaningAction(action="drop_column", column=col, reason=issue))
            elif issue == "numeric_stored_as_text":
                add(col, issue, "medium", "convert to numbers", "at least 95% of the values parse as numbers",
                    0.95, "rule", ev, info, CleaningAction(action="to_numeric", column=col, reason="numbers stored as text"))
            elif issue == "datetime_stored_as_text":
                add(col, issue, "medium", "parse as dates", "at least 90% of the values parse as dates",
                    0.9, "rule", ev, info, CleaningAction(action="to_datetime", column=col, reason="dates stored as text"))
            elif issue == "high_cardinality":
                add(col, issue, "medium", "target or frequency encoding fitted inside the CV folds",
                    f"{info['n_unique']} levels: one-hot encoding would explode the feature space",
                    clip01(info["n_unique"] / 100), "rule", ev, info,
                    CleaningAction(action="high_card_encoding", column=col, stage="pipeline", reason="many levels"))
            elif issue == "free_text":
                add(col, issue, "low", "drop the column (free text is not modelled in this version)",
                    "long multi-word values", 0.8, "rule", ev, info,
                    CleaningAction(action="drop_column", column=col, reason="free text"))

        # ---- missing values
        miss, ev = self.ctx.tool("pandas_tools.missing_report", pt.missing_report, {"target": target}, df, target)
        for col, r in miss["columns"].items():
            pct, mech = r["missing_pct"], r["mechanism"]
            if col == target:
                add(col, "missing_target", "high", "drop rows with a missing target",
                    f"{pct:.1f}% of target values are missing; these rows cannot be used for training", 1.0, "rule", ev, r,
                    CleaningAction(action="drop_rows_missing_target", column=col, reason="missing target"))
                continue
            if col in protected:
                continue
            if mech == "entirely missing" or pct > 60:
                add(col, "missing_values", "high", "drop the column", f"{pct:.1f}% missing ({mech})", r["confidence"], "effect_size", ev, r,
                    CleaningAction(action="drop_column", column=col, reason=f"{pct:.0f}% missing"))
                continue
            sev = "high" if pct > 40 else "medium" if (pct > 5 or mech != "MCAR-like") else "low"
            assoc = r.get("associations") or {}
            why = f"{pct:.1f}% missing; mechanism looks {mech}"
            if mech == "MAR-like" and assoc.get("strongest_with"):
                why += f" (missingness tracks '{assoc['strongest_with']}', effect {assoc['effect']:.2f})"
            if mech == "possibly MNAR":
                why += (f" (rows with a missing value are predicted to sit in the {r['tail']} tail, d = {r['self_effect_d']:.2f}; "
                        "this cannot be confirmed from the data)")
            action = "impute inside the CV pipeline" + (" and add a missing indicator" if mech != "MCAR-like" else "")
            add(col, "missing_values", sev, action, why, r["confidence"], "effect_size", ev, r,
                CleaningAction(action="impute", column=col, stage="pipeline",
                               params={"add_indicator": mech != "MCAR-like"}, reason=mech))

        # ---- outliers
        out, ev = self.ctx.tool("pandas_tools.outlier_report", pt.outlier_report, {"exclude": sorted(protected)},
                                df, sorted(protected), self.ctx.seed)
        for col, r in out["columns"].items():
            pct, skew = r["iqr_pct"], r.get("skew") or 0
            if pct > 10 and abs(skew) > 1:
                add(col, "heavy_tail", "low", "power transform inside the pipeline (for linear models)",
                    f"{pct:.1f}% of values beyond the IQR fences and skew {skew:.2f}: a heavy tail, not isolated outliers",
                    r["agreement"], "rule", ev, r, CleaningAction(action="power_transform", column=col, stage="pipeline", reason="heavy tail"))
            elif pct >= 0.5:
                sev = "medium" if pct > 2 else "low"
                add(col, "outliers", sev, "cap at IQR fences fitted inside each CV fold",
                    f"{r['iqr_count']} values ({pct:.1f}%) beyond the IQR fences; z-score and Isolation Forest agree on "
                    f"{r['agreement']:.0%}", r["agreement"], "rule", ev, r,
                    CleaningAction(action="winsorize", column=col, stage="pipeline", reason="outliers"))

        # ---- duplicates
        dup, ev = self.ctx.tool("pandas_tools.duplicate_report", pt.duplicate_report, {}, df)
        if dup["exact_count"]:
            sev = "high" if dup["exact_pct"] > 1 else "medium"
            add(None, "duplicates", sev, "drop exact duplicate rows before the train/test split",
                f"{dup['exact_count']} exact duplicate rows ({dup['exact_pct']:.1f}%): copies can land in both train and test "
                "and inflate the score", 1.0, "rule", ev, None, CleaningAction(action="drop_duplicates", reason="exact duplicates"))
        if dup["near_count"]:
            add(None, "near_duplicates", "low", "review; only exact copies are dropped automatically",
                f"{dup['near_count']} rows match another row after ignoring case, spacing and identifiers", 0.6, "rule", ev, None)

        # ---- class imbalance
        if fam == "classification" and target:
            imb, ev = self.ctx.tool("pandas_tools.imbalance_report", pt.imbalance_report, {"target": target}, df, target)
            ms = imb["minority_share"]
            if ms < 0.2 or imb["imbalance_ratio"] > 5:
                sev = "high" if ms < 0.05 else "medium"
                add(target, "class_imbalance", sev, "class weights inside the pipeline; primary metric is PR-AUC",
                    f"minority class '{imb['minority_class']}' is {ms:.1%} of rows (ratio {imb['imbalance_ratio']:.1f}:1)",
                    clip01(1 - ms / 0.2), "effect_size", ev, imb,
                    CleaningAction(action="class_weight", column=target, stage="pipeline", reason="imbalance"))

        # ---- leakage (the most important tool)
        if target and fam in ("classification", "regression", "anomaly_detection"):
            leak, ev = self.ctx.tool("pandas_tools.leakage_report", pt.leakage_report, {"target": target, "task": fam},
                                     df, target, fam, self.ctx.seed)
            for col in leak["flagged"]:
                r = leak["features"][col]
                if col in protected:
                    continue
                sev, strength = r["severity"], r["strength"]
                if sev == "blocker":
                    add(col, "target_leakage", "blocker", "confirm dropping the column, or explain why it is available at prediction time",
                        f"'{col}' alone predicts the target almost perfectly (strength {strength:.3f})"
                        + (f"; its name suggests '{r['name_flag']}'" if r["name_flag"] else ""),
                        strength, "effect_size", ev, r, CleaningAction(action="drop_column", column=col, reason="target leakage"))
                elif sev == "high":
                    add(col, "possible_proxy", "high", "investigate: drop it if it is recorded after the outcome",
                        f"'{col}' is a very strong single predictor (strength {strength:.3f}); it may be a proxy for the target",
                        strength, "effect_size", ev, r,
                        CleaningAction(action="drop_column", column=col, reason="possible proxy",
                                       enabled=bool(r["name_flag"])))
                else:
                    add(col, "suspicious_name", "medium", "check when this column is recorded",
                        f"the name suggests information from after the outcome ('{r['name_flag']}'); strength {strength:.3f}",
                        strength, "effect_size", ev, r)

        # ---- inconsistencies
        inc, ev = self.ctx.tool("pandas_tools.inconsistency_report", pt.inconsistency_report, {}, df)
        for col, r in inc["columns"].items():
            if "case_variants" in r:
                add(col, "mixed_case", "medium", "lower-case and strip the categories",
                    f"{r['case_variant_groups']} categories appear in several spellings, e.g. {list(r['case_variants'].values())[0]}",
                    1.0, "rule", ev, r, CleaningAction(action="recode_case", column=col, reason="mixed case"))
            if "negative_values" in r:
                add(col, "impossible_values", "medium", "set negative values to missing",
                    f"{r['negative_values']} negative values in a column that cannot be negative", 0.9, "rule", ev, r,
                    CleaningAction(action="set_nan_invalid", column=col, params={"min": 0}, reason="impossible negatives"))
            if "impossible_age_over_120" in r:
                add(col, "impossible_values", "medium", "set ages over 120 to missing",
                    f"{r['impossible_age_over_120']} ages above 120", 0.9, "rule", ev, r,
                    CleaningAction(action="set_nan_invalid", column=col, params={"max": 120}, reason="impossible ages"))
            if "mixed_units" in r:
                add(col, "mixed_units", "high", "fix units by hand before modelling",
                    f"values carry different units: {r['mixed_units']}", 0.8, "rule", ev, r)

        # ---- label noise (classification only)
        if fam == "classification" and target and len(df) >= 50:
            ln, ev = self.ctx.tool("stats_tools.label_noise_report", stt.label_noise_report, {"target": target}, df, target, self.ctx.seed)
            rate = ln.get("estimated_noise_rate")
            if rate is not None and rate >= 0.05:
                add(target, "label_noise", "medium", "review the suspect rows; no automatic relabelling",
                    f"out-of-fold predictions confidently disagree with {ln['suspect_count']} labels ({rate:.1%})",
                    clip01(rate / 0.2), "rule", ev, ln)

        findings.sort(key=lambda f: SEVERITY_ORDER[f.severity])
        blockers = [f.column for f in findings if f.severity == "blocker" and f.column]
        report = DataAuditReport(findings=findings, cleaning_plan=self._dedupe(actions), blockers=blockers,
                                 summary=self._summary(findings))
        self._log(report)
        self.finished(report.summary)
        return report

    @staticmethod
    def _dedupe(actions: list[CleaningAction]) -> list[CleaningAction]:
        seen, out = set(), []
        for a in actions:
            key = (a.action, a.column)
            if key not in seen:
                seen.add(key)
                out.append(a)
        drop_cols = {a.column for a in out if a.action == "drop_column" and a.enabled}
        out = [a for a in out if a.action == "drop_column" or a.column not in drop_cols]
        # exact duplicates are defined on the raw rows, so they are removed before anything else
        return sorted(out, key=lambda a: ACTION_ORDER.get(a.action, 50))

    @staticmethod
    def _summary(findings: list[AuditFinding]) -> str:
        if not findings:
            return "No data-quality issues found."
        counts = {s: sum(1 for f in findings if f.severity == s) for s in SEVERITY_ORDER}
        parts = [f"{n} {s}" for s, n in counts.items() if n]
        return f"{len(findings)} findings ({', '.join(parts)})."

    def _log(self, report: DataAuditReport) -> None:
        for f in report.findings:
            if f.severity in ("blocker", "high", "medium"):
                self.record(f"audit:{f.issue_type}" + (f":{f.column}" if f.column else ""), f.recommended_action,
                            f.rationale, f.evidence, f.confidence, f.confidence_method,
                            options=[f.recommended_action, "leave as is"], chosen="proposed: " + f.recommended_action,
                            note="proposal only; applied after approval" if f.severity != "blocker" else "blocker: run pauses for the client")
