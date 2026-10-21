"""Statistician: answers "can we actually say that?" and produces the profile that feeds the
dataset fingerprint, the drift check and the guidance for downstream agents."""
from __future__ import annotations

import pandas as pd

from agents.base import BaseAgent, cv_claim, slice_evidence
from app.schemas import DataAuditReport, DebatePosition, Evidence, FeatureProfile, StatFlag, StatisticalProfile, TaskPlan
from tools import fingerprint_tools as ft
from tools import stats_tools as stt
from tools.common import clip01

VALIDITY_CHECKS = {"overfit_check", "seed_stability", "subgroup_check", "leakage_probe", "baseline_check"}
SEVERITY_WEIGHT = {"medium": 0.7, "high": 0.85, "blocker": 0.95}


def validity_issues(cand) -> list:
    return [o for o in cand.objections if not o.resolved and o.check in VALIDITY_CHECKS and o.severity in SEVERITY_WEIGHT]


class StatisticianAgent(BaseAgent):
    name = "statistician"

    def profile(self, df: pd.DataFrame, plan: TaskPlan, audit: DataAuditReport | None) -> StatisticalProfile:
        self.started("testing assumptions and computing effect sizes")
        target, fam = plan.target, plan.task_family
        exclude = {c for c in (target, plan.time_column, plan.action_column, plan.reward_column,
                               plan.propensity_column, plan.group_column) if c}
        num = stt.numeric_features(df, exclude)
        cat = stt.categorical_features(df, exclude)
        tool = self.ctx.tool
        flags: list[StatFlag] = []
        guidance: list[str] = []
        evs: list[Evidence] = []

        norm, ev_n = tool("stats_tools.normality_tests", stt.normality_tests, {"columns": num}, df, num)
        dist, ev_d = tool("stats_tools.distribution_fit", stt.distribution_fit, {"columns": num}, df, num, self.ctx.seed)
        corr, ev_c = tool("stats_tools.correlation_report", stt.correlation_report, {"columns": num}, df, num)
        vif, ev_v = tool("stats_tools.vif_report", stt.vif_report, {"columns": num}, df, num)
        evs += [ev_n, ev_d, ev_c, ev_v]
        eff: dict = {"columns": {}}
        supervised = fam in ("classification", "regression") and target
        if supervised:
            eff, ev_e = tool("stats_tools.effect_sizes", stt.effect_sizes, {"target": target, "task": fam},
                             df, target, fam, num, cat)
            homo, ev_h = tool("stats_tools.homoscedasticity_test", stt.homoscedasticity_test,
                              {"target": target, "task": fam}, df, target, fam, num)
            evs += [ev_e, ev_h]
            if fam == "regression" and homo.get("heteroscedastic"):
                flags.append(StatFlag(issue="heteroscedasticity", features=[target], severity="medium",
                                      detail=f"Breusch-Pagan p = {homo['p_value']:.2g}: error variance changes with the features",
                                      confidence=0.8 if homo["p_value"] < 0.001 else 0.65, confidence_method="rule",
                                      evidence=[ev_h]))
                guidance.append("Error variance is not constant: linear-model standard errors are unreliable; "
                                "prefer tree models or a transformed target, and report intervals with care.")
            if fam == "classification":
                unequal = [c for c, r in homo.get("columns", {}).items() if not r["equal_variance"]]
                if unequal:
                    flags.append(StatFlag(issue="unequal_variance_across_classes", features=unequal[:10], severity="low",
                                          detail=f"Levene p < 0.01 for {len(unequal)} feature(s)", confidence=0.6,
                                          confidence_method="rule", evidence=[ev_h]))
        if fam in ("regression", "time_series") and target:
            ind, ev_i = tool("stats_tools.independence_check", stt.independence_check,
                             {"target": target, "time_column": plan.time_column}, df, target, num, plan.time_column)
            evs.append(ev_i)
            if ind.get("autocorrelated") and fam == "regression":
                flags.append(StatFlag(issue="autocorrelation", features=[target], severity="medium",
                                      detail=f"Durbin-Watson {ind['durbin_watson']:.2f} in {ind['ordered_by']}",
                                      confidence=clip01(abs(2 - ind["durbin_watson"]) / 2 + 0.3), confidence_method="effect_size",
                                      evidence=[ev_i]))
                guidance.append("Rows are not independent in their stored order: use grouped or time-ordered splits "
                                "if the order means time.")

        # ---- per-feature profile
        feats = []
        for c in num + cat:
            n_r = norm["columns"].get(c, {})
            d_r = dist["columns"].get(c, {})
            e_r = eff["columns"].get(c, {})
            s = df[c]
            feats.append(FeatureProfile(
                name=c, kind="numeric" if c in num else "categorical",
                mean=float(s.mean()) if c in num else None, std=float(s.std()) if c in num else None,
                skew=n_r.get("skew"), kurtosis=n_r.get("kurtosis"), normality_test=n_r.get("test"),
                normal_p=n_r.get("p_value"), best_fit=d_r.get("best"), best_fit_ks=d_r.get("ks"),
                effect_measure=e_r.get("measure"), effect_size=e_r.get("effect"), p_value=e_r.get("p_value")))

        # ---- flags and guidance
        skewed = [c for c, r in norm["columns"].items() if abs(r["skew"]) > 1]
        if skewed:
            top = max(abs(norm["columns"][c]["skew"]) for c in skewed)
            flags.append(StatFlag(issue="heavy_skew", features=skewed, severity="low",
                                  detail=f"{len(skewed)} feature(s) with |skew| > 1 (max {top:.2f})",
                                  confidence=clip01(0.5 + (top - 1) / 4), confidence_method="effect_size",
                                  evidence=[slice_evidence(ev_n, skewed[0], norm["columns"][skewed[0]])]))
            guidance.append(f"Heavy skew in {', '.join(skewed[:5])}: add a power transform for linear and distance-based models.")
        severe = [c for c, r in vif.get("columns", {}).items() if r["vif"] > 10]
        warn = [c for c, r in vif.get("columns", {}).items() if 5 < r["vif"] <= 10]
        if severe or warn:
            worst = max(vif["columns"][c]["vif"] for c in severe + warn)
            pair = corr["top_pairs"][0] if corr.get("top_pairs") else None
            detail = f"VIF above 10 for {len(severe)} and above 5 for {len(warn)} feature(s); max {min(worst, 1e6):.1f}"
            flags.append(StatFlag(issue="multicollinearity", features=severe + warn, severity="high" if severe else "medium",
                                  detail=detail, confidence=clip01(0.5 + (min(worst, 50) - 5) / 20),
                                  confidence_method="effect_size", evidence=[ev_v]))
            guidance.append("High VIF" + (f" between {pair['a']} and {pair['b']} (r = {pair['pearson']:.2f})" if pair else "")
                            + ": prefer regularized models (ridge or lasso); do not read single linear coefficients.")
        n, p = len(df), len(num) + len(cat)
        if n < 300:
            flags.append(StatFlag(issue="small_sample", features=[], severity="medium", detail=f"only {n} rows",
                                  confidence=clip01(1 - n / 300 + 0.5), confidence_method="rule", evidence=[ev_n]))
            guidance.append(f"Only {n} rows: cap model complexity and expect wide confidence intervals.")
        if n and p / n > 0.1:
            guidance.append(f"{p} features for {n} rows: regularize and consider feature selection.")
        if supervised and eff["columns"]:
            strongest = max(eff["columns"].items(), key=lambda kv: kv[1]["effect"])
            weak = all(r["effect"] < (0.2 if r["measure"] == "cohens_d" else 0.1) for r in eff["columns"].values())
            if weak:
                flags.append(StatFlag(issue="weak_signal", features=[strongest[0]], severity="medium",
                                      detail=f"strongest single effect is {strongest[1]['measure']} = {strongest[1]['effect']:.3f}",
                                      confidence=0.7, confidence_method="effect_size",
                                      evidence=[slice_evidence(ev_e, strongest[0], strongest[1])]))
                guidance.append("Every single-feature effect is small: expect modest gains over the baseline and "
                                "treat any large improvement with suspicion.")
            else:
                guidance.append(f"Strongest single predictor: {strongest[0]} ({strongest[1]['measure']} = "
                                f"{strongest[1]['effect']:.3f}, p = {strongest[1]['p_value']:.2g}).")

        # ---- signals for the Architect, from audit + statistics
        plan_actions = {a.action for a in (audit.cleaning_plan if audit else []) if a.enabled}
        imbalance = None
        if fam == "classification" and target:
            vc = df[target].astype(str).value_counts()
            imbalance = float(vc.max() / vc.min())
        signals = {
            "n_rows": n, "n_features": p, "small_sample": n < 300,
            "high_vif": bool(severe), "vif_features": severe + warn, "mean_vif": vif.get("mean_vif", 1.0),
            "heavy_skew_features": skewed, "heavy_skew": bool(skewed),
            "heteroscedastic": any(f.issue == "heteroscedasticity" for f in flags),
            "autocorrelated": any(f.issue == "autocorrelation" for f in flags),
            "weak_signal": any(f.issue == "weak_signal" for f in flags),
            "imbalance_ratio": imbalance, "class_weight": "class_weight" in plan_actions,
            "winsorize": "winsorize" in plan_actions, "power_transform": "power_transform" in plan_actions or bool(skewed),
            "add_indicator": any(a.action == "impute" and a.params.get("add_indicator") for a in (audit.cleaning_plan if audit else [])),
            "high_cardinality": [a.column for a in (audit.cleaning_plan if audit else []) if a.action == "high_card_encoding"],
            "mean_abs_corr": corr.get("mean_abs_corr", 0.0),
        }
        fp, ev_f = tool("fingerprint_tools.dataset_fingerprint", ft.dataset_fingerprint, {"task": fam},
                        df, fam, target, imbalance, corr.get("mean_abs_corr", 0.0), vif.get("mean_vif", 1.0))
        evs.append(ev_f)
        profile = StatisticalProfile(features=feats, flags=flags, guidance=guidance, signals=signals,
                                     top_correlations=corr.get("top_pairs", []), fingerprint=fp["vector"],
                                     fingerprint_features=fp["features"])
        conf = max([f.confidence for f in flags], default=0.9)
        self.record("statistical_guidance", "; ".join(guidance) or "no assumption violations flagged",
                    f"{len(flags)} flag(s): " + ", ".join(f.issue for f in flags) if flags else "assumption checks passed",
                    evs, conf, "effect_size" if flags else "rule", options=["guidance"], chosen="guidance issued")
        self.finished(f"{len(flags)} flags, fingerprint computed")
        return profile

    def position(self, survivors, plan, round_no: int = 0):
        """Prefer the best-scoring candidate with no unresolved validity issue."""
        from agents.ml_architect.core import MLArchitectAgent  # local import avoids a cycle
        ranked = MLArchitectAgent.rank(survivors, plan.metric)
        clean = [c for c in ranked if not validity_issues(c)]
        if clean:
            pref = clean[0]
        else:
            pref = min(ranked, key=lambda c: max(SEVERITY_WEIGHT[o.severity] for o in validity_issues(c)))
        top = ranked[0]
        reasons = []
        if pref.id != top.id:
            for o in validity_issues(top):
                reasons.append(o.claim.model_copy(deep=True))
            conf = max(SEVERITY_WEIGHT[o.severity] for o in validity_issues(top))
        else:
            conf = 0.6
        c = cv_claim(self.ctx.log, pref, f"{pref.id} passes the validity checks and scores {pref.cv.primary_mean:.4f} "
                                            f"{pref.cv.primary}" if not validity_issues(pref) else None)
        if c:
            reasons.append(c)
        return DebatePosition(agent=self.name, preferred=pref.id, reasons=reasons, confidence=conf,
                              confidence_method="rule", round=round_no)
