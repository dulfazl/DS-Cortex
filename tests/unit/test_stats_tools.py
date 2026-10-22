"""Known-answer tests: each tool is compared with a direct scipy/statsmodels call or a value we
can compute by hand. "The LLM never invents numbers" only helps if the tools are right."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from tools import drift_check_tools as dc
from tools import stats_tools as stt
from tools.fingerprint_tools import dataset_fingerprint, fingerprint_distance


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 500
    a = rng.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": a * 2 + rng.normal(0, 0.01, n), "c": rng.normal(5, 2, n),
                         "skewed": rng.lognormal(0, 1, n), "cat": rng.choice(["x", "y", "z"], n),
                         "y": (a + rng.normal(0, 1, n) > 0).astype(int)})


def test_vif_collinear_is_severe(df):
    r = stt.vif_report(df, ["a", "b", "c"])
    assert r["columns"]["a"]["vif"] > 100 and r["columns"]["a"]["level"] == "severe"
    assert r["columns"]["c"]["vif"] < 1.5


def test_normality_matches_scipy(df):
    r = stt.normality_tests(df, ["c", "skewed"])
    assert r["columns"]["c"]["p_value"] == pytest.approx(st.shapiro(df["c"]).pvalue, rel=1e-6)
    assert r["columns"]["skewed"]["normal"] is False
    assert r["columns"]["skewed"]["skew"] == pytest.approx(st.skew(df["skewed"]), rel=1e-6)


def test_cramers_v_of_column_with_itself_is_one(df):
    d = df.assign(y=df["cat"])
    r = stt.effect_sizes(d, "y", "classification", [], ["cat"])
    assert r["columns"]["cat"]["effect"] == pytest.approx(1.0, abs=1e-6)


def test_cohens_d_by_hand(df):
    r = stt.effect_sizes(df, "y", "classification", ["a"], [])
    g1, g0 = df.loc[df.y == 1, "a"], df.loc[df.y == 0, "a"]
    pooled = np.sqrt(((len(g1) - 1) * g1.var() + (len(g0) - 1) * g0.var()) / (len(g1) + len(g0) - 2))
    assert r["columns"]["a"]["effect"] == pytest.approx(abs(g1.mean() - g0.mean()) / pooled, rel=1e-6)


def test_distribution_fit_picks_lognormal(df):
    r = stt.distribution_fit(df, ["skewed", "c"])
    assert r["columns"]["skewed"]["best"] == "lognorm"
    assert r["columns"]["c"]["best"] == "norm"


def test_correlation_top_pair(df):
    r = stt.correlation_report(df, ["a", "b", "c"])
    assert {r["top_pairs"][0]["a"], r["top_pairs"][0]["b"]} == {"a", "b"}
    assert r["top_pairs"][0]["pearson"] == pytest.approx(df["a"].corr(df["b"]), rel=1e-5)


def test_drift_flags_exactly_the_shifted_feature(df):
    first, second = df.iloc[:250].copy(), df.iloc[250:].copy()
    second["c"] = second["c"] + 3
    r = dc.drift_check(first, second, exclude=["y"])
    assert r["drifted"] == ["c"]
    assert r["drift_score"] == pytest.approx(1 / 5)
    assert r["features"]["c"]["statistic"] == pytest.approx(st.ks_2samp(first["c"], second["c"]).statistic)
    assert "does not retrain" in r["limits"]


def test_identical_samples_ks_zero(df):
    r = dc.drift_check(df, df.copy(), exclude=["y"])
    assert r["drifted"] == [] and r["features"]["a"]["statistic"] == 0


def test_fingerprint_fixed_length_and_distance(df):
    f1 = dataset_fingerprint(df, "classification", "y", 1.1, 0.3, 4.0)
    f2 = dataset_fingerprint(df.sample(frac=0.5, random_state=0), "classification", "y", 1.1, 0.3, 4.0)
    f3 = dataset_fingerprint(df, "regression", "c", None, 0.3, 4.0)
    assert len(f1["vector"]) == 12
    assert fingerprint_distance(f1["vector"], f2["vector"]) < fingerprint_distance(f1["vector"], f3["vector"])


def test_label_noise_on_a_capped_sample_reports_file_rows():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    y = np.where(x > 0, "a", "b")
    flipped = rng.choice(n, 60, replace=False)
    y[flipped] = np.where(y[flipped] == "a", "b", "a")
    df = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": y})
    full = stt.label_noise_report(df, "y", seed=0)
    capped = stt.label_noise_report(df, "y", seed=0, max_rows=1000)
    assert full["rows_checked"] == n and capped["rows_checked"] == 1000
    assert 0.01 < capped["estimated_noise_rate"] < 0.05     # 2% of labels were flipped
    found = set(capped["suspect_rows"])
    assert found and len(found & set(flipped.tolist())) / len(found) > 0.8   # row numbers refer to the file


def _preds(y, pred, proba=None, classes=None):
    return {"y_true": list(y), "y_pred": list(pred), "proba": proba, "classes": classes or [], "pos": None}


def test_holdout_gain_separates_noise_from_signal():
    from tools import sklearn_tools as sk
    rng = np.random.default_rng(0)
    y = rng.choice(4, 400, p=[.5, .17, .17, .16])
    prior = np.tile([.5, .17, .17, .16], (400, 1))
    noise = rng.dirichlet(np.ones(4), 400)
    signal = np.eye(4)[y] * 0.5 + noise * 0.5
    base = _preds(y, prior.argmax(1), prior.tolist(), list("abcd"))
    r = sk.holdout_gain(_preds(y, noise.argmax(1), noise.tolist(), list("abcd")), base, "pr_auc", 0, 300)
    assert r["applicable"] and r["meaningful"] is False and r["ci95"][0] < 0 < r["ci95"][1]
    r = sk.holdout_gain(_preds(y, signal.argmax(1), signal.tolist(), list("abcd")), base, "pr_auc", 0, 300)
    assert r["meaningful"] is True and r["ci95"][0] > 0


def test_holdout_gain_is_oriented_for_error_metrics():
    from tools import sklearn_tools as sk
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    good = sk.holdout_gain(_preds(y, y + rng.normal(0, 0.1, 300)), _preds(y, np.zeros(300)), "rmse", 0, 300)
    assert good["gain"] > 0 and good["meaningful"]          # lower rmse than the baseline is a positive gain
    assert sk.holdout_gain(_preds(y[:10], y[:10]), _preds(y[:10], y[:10]), "rmse", 0)["applicable"] is False
