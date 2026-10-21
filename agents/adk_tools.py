"""Path-based tool functions for the ADK agents (`adk web agents`), so each agent can be explored
interactively against a CSV file. The workflow itself calls the deterministic agent classes."""
from __future__ import annotations

from tools import pandas_tools as pt
from tools import router_tools as rt
from tools import stats_tools as stt
from tools.common import jsonable


def _df(path: str):
    return pt.coerce_types(pt.load_csv(path))


def profile_and_precheck(path: str, problem_statement: str) -> dict:
    """Quick-profile the CSV at `path` and run the rule-based routing pre-checks for the statement."""
    profile = pt.quick_profile(path)
    return jsonable({"profile": profile, "precheck": rt.precheck(problem_statement, profile, _df(path)),
                     "metric_table": rt.METRICS})


def audit_csv(path: str, target: str) -> dict:
    """Run the data-quality tools (missing values, types, outliers, duplicates, leakage, inconsistencies)."""
    df = _df(path)
    out = {"missing": pt.missing_report(df, target), "types": pt.type_inference(df),
           "outliers": pt.outlier_report(df, [target]), "duplicates": pt.duplicate_report(df),
           "inconsistencies": pt.inconsistency_report(df)}
    if target in df:
        fam = "classification" if df[target].nunique() <= 20 else "regression"
        out["leakage"] = pt.leakage_report(df, target, fam)
    return jsonable(out)


def statistics_csv(path: str, target: str) -> dict:
    """Normality, VIF, correlations and effect sizes for the numeric and categorical features."""
    df = _df(path)
    num = stt.numeric_features(df, {target})
    cat = stt.categorical_features(df, {target})
    fam = "classification" if df[target].nunique() <= 20 else "regression"
    return jsonable({"normality": stt.normality_tests(df, num), "vif": stt.vif_report(df, num),
                     "correlations": stt.correlation_report(df, num),
                     "effect_sizes": stt.effect_sizes(df, target, fam, num, cat)})
