"""Plot builders. Each returns Plotly JSON (a dict), never an image, so the dashboard renders it
client-side. Only plots relevant to the stated problem are made, ranked by relevance."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

KEYWORD_BOOST = {
    "feature_importance": ["why", "driver", "explain", "important", "factor", "reason", "influence"],
    "confusion_matrix": ["mistake", "error", "wrong", "misclass", "confus"],
    "pr_curve": ["fraud", "rare", "imbalanc", "precision", "recall"],
    "calibration_curve": ["probabilit", "risk score", "calibrat", "confiden"],
    "residuals_vs_predicted": ["error", "residual", "accuracy"],
    "forecast": ["forecast", "next", "future", "predict"],
    "cluster_projection": ["segment", "group", "cluster", "persona"],
    "score_histogram": ["unusual", "anomal", "fraud", "outlier"],
    "missing_map": ["missing", "quality", "clean"],
    "correlation_heatmap": ["relationship", "correlat", "related"],
}


def _fig(fig: go.Figure, title: str) -> dict:
    fig.update_layout(title=title, margin=dict(l=50, r=20, t=50, b=45), height=360)
    out = json.loads(fig.to_json())
    out.get("layout", {}).pop("template", None)
    return out


def _plot(pid: str, title: str, group: str, base: float, fig: go.Figure, caption: str) -> dict:
    return {"id": pid, "title": title, "group": group, "relevance": base, "caption": caption, "figure": _fig(fig, title)}


def overview_plots(df: pd.DataFrame, target: str | None, corr: dict | None, top_features: list[str]) -> list[dict]:
    plots = []
    cols = list(df.columns[:50])
    sample = df[cols].sample(min(len(df), 200), random_state=0) if len(df) > 200 else df[cols]
    miss = sample.isna().astype(int).values
    if miss.any():
        plots.append(_plot("missing_map", "Missing-value map", "data", 0.35,
                           go.Figure(go.Heatmap(z=miss.T, y=cols, colorscale=[[0, "#e8eef7"], [1, "#d1495b"]], showscale=False)),
                           "Each row of the heatmap is a column; red cells are missing values in a sample of rows."))
    for c in top_features[:4]:
        if c in df and pd.api.types.is_numeric_dtype(df[c]):
            plots.append(_plot(f"dist_{c}", f"Distribution of {c}", "data", 0.3,
                               go.Figure(go.Histogram(x=df[c].dropna().values, nbinsx=40)),
                               f"Histogram of {c}, one of the strongest single predictors."))
    if corr and corr.get("matrix") and len(corr["matrix"]["columns"]) >= 2:
        m = corr["matrix"]
        plots.append(_plot("correlation_heatmap", "Correlation heatmap (Pearson)", "data", 0.3,
                           go.Figure(go.Heatmap(z=m["values"], x=m["columns"], y=m["columns"], zmin=-1, zmax=1,
                                                colorscale="RdBu", reversescale=True)),
                           "Pairwise Pearson correlations between numeric features."))
    if target and target in df:
        y = df[target].dropna()
        trace = go.Histogram(x=y.values, nbinsx=40) if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10 else \
            go.Bar(x=[str(k) for k in y.value_counts().index], y=y.value_counts().values)
        plots.append(_plot("target_distribution", f"Target distribution: {target}", "data", 0.45, go.Figure(trace),
                           f"How the target '{target}' is distributed."))
    return plots


def classification_plots(err: dict) -> list[dict]:
    plots = []
    cls = [str(c) for c in err["classes"]]
    plots.append(_plot("confusion_matrix", "Confusion matrix (held-out test set)", "model", 0.8,
                       go.Figure(go.Heatmap(z=err["confusion_matrix"], x=[f"pred {c}" for c in cls], y=[f"true {c}" for c in cls],
                                            text=err["confusion_matrix"], texttemplate="%{text}", colorscale="Blues")),
                       "Rows are true classes, columns are predictions, on rows never used for training or selection."))
    if "roc" in err:
        f = go.Figure(go.Scatter(x=err["roc"]["fpr"], y=err["roc"]["tpr"], mode="lines", name="model"))
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="chance"))
        f.update_xaxes(title="false positive rate")
        f.update_yaxes(title="true positive rate")
        plots.append(_plot("roc_curve", "ROC curve", "model", 0.6, f, f"Positive class: {err['positive_class']}."))
        f = go.Figure(go.Scatter(x=err["pr"]["recall"], y=err["pr"]["precision"], mode="lines"))
        f.update_xaxes(title="recall")
        f.update_yaxes(title="precision")
        plots.append(_plot("pr_curve", "Precision-recall curve", "model", 0.6, f, "Precision against recall as the threshold moves."))
        f = go.Figure(go.Scatter(x=err["calibration"]["mean_predicted"], y=err["calibration"]["fraction_positive"],
                                 mode="lines+markers", name="model"))
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="perfectly calibrated"))
        f.update_xaxes(title="mean predicted probability")
        f.update_yaxes(title="observed fraction positive")
        plots.append(_plot("calibration_curve", "Calibration curve", "model", 0.5, f,
                           "Points on the diagonal mean predicted probabilities match observed frequencies."))
    return plots


def regression_plots(err: dict) -> list[dict]:
    y, yp = np.asarray(err["y_true"]), np.asarray(err["y_pred"])
    lo, hi = float(min(y.min(), yp.min())), float(max(y.max(), yp.max()))
    f = go.Figure(go.Scatter(x=y, y=yp, mode="markers", marker=dict(size=5, opacity=0.6), name="test rows"))
    f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash"), name="perfect"))
    f.update_xaxes(title="actual")
    f.update_yaxes(title="predicted")
    g = go.Figure(go.Scatter(x=yp, y=y - yp, mode="markers", marker=dict(size=5, opacity=0.6)))
    g.add_hline(y=0, line_dash="dash")
    g.update_xaxes(title="predicted")
    g.update_yaxes(title="residual (actual - predicted)")
    return [_plot("predicted_vs_actual", "Predicted vs actual (held-out test set)", "model", 0.8, f,
                  "Points near the diagonal are accurate predictions."),
            _plot("residuals_vs_predicted", "Residuals vs predicted", "model", 0.6, g,
                  "A funnel shape means errors grow with the prediction (heteroscedasticity).")]


def clustering_plots(prof: dict) -> list[dict]:
    plots = []
    pca = prof["pca"]
    f = go.Figure(go.Scatter(x=pca["x"], y=pca["y"], mode="markers",
                             marker=dict(color=pca["label"], colorscale="Viridis", size=6, opacity=0.7),
                             text=[f"cluster {lab}" for lab in pca["label"]]))
    f.update_xaxes(title="PC 1")
    f.update_yaxes(title="PC 2")
    plots.append(_plot("cluster_projection", "Clusters in a 2D PCA projection", "model", 0.85, f,
                       "Each point is a row, coloured by its cluster."))
    if prof["silhouette_samples"]:
        s, lab = np.asarray(prof["silhouette_samples"]), np.asarray(prof["silhouette_labels"])
        order = np.lexsort((-s, lab))
        plots.append(_plot("silhouette_plot", "Silhouette plot", "model", 0.6,
                           go.Figure(go.Bar(x=s[order], y=list(range(len(order))), orientation="h",
                                            marker=dict(color=lab[order], colorscale="Viridis"))),
                           "Bars above zero sit closer to their own cluster than to the next one."))
    if prof["profiles"]:
        f = go.Figure()
        for k, row in prof["profiles"].items():
            f.add_trace(go.Bar(name=f"cluster {k}", x=list(row.keys()), y=list(row.values())))
        f.update_layout(barmode="group")
        plots.append(_plot("cluster_profiles", "Cluster profiles (feature means)", "model", 0.7, f,
                           "Mean of each feature per cluster."))
    return plots


def anomaly_plots(err: dict) -> list[dict]:
    f = go.Figure(go.Histogram(x=err["scores"], nbinsx=50))
    f.add_vline(x=err["threshold"], line_dash="dash", annotation_text="top 5% threshold")
    f.update_xaxes(title="anomaly score (higher = more unusual)")
    return [_plot("score_histogram", "Anomaly score histogram", "model", 0.85, f,
                  f"{err['flagged']} held-out rows score above the threshold.")]


def time_series_plots(err: dict) -> list[dict]:
    dates, y, n_test = err["dates"], err["y"], err["n_test"]
    fc = err["forecast"]
    f = go.Figure(go.Scatter(x=dates, y=y, mode="lines", name="actual"))
    test_dates = dates[-n_test:]
    if err.get("half_width"):
        hw = np.asarray(err["half_width"])
        f.add_trace(go.Scatter(x=test_dates + test_dates[::-1], y=list(np.asarray(fc) + hw) + list((np.asarray(fc) - hw)[::-1]),
                               fill="toself", line=dict(width=0), opacity=0.25, name="95% interval"))
    f.add_trace(go.Scatter(x=test_dates, y=fc, mode="lines", name="forecast", line=dict(dash="dash")))
    plots = [_plot("forecast", "Forecast with interval (held-out period)", "model", 0.9, f,
                   "The dashed line is the forecast for the held-out period; the band is a 95% interval.")]
    m = int(err.get("season") or 1)
    if m > 1 and len(y) >= 2 * m:
        from statsmodels.tsa.seasonal import seasonal_decompose
        dec = seasonal_decompose(np.asarray(y, float), period=m, extrapolate_trend="freq")
        g = go.Figure()
        for name, comp in (("trend", dec.trend), ("seasonal", dec.seasonal), ("residual", dec.resid)):
            g.add_trace(go.Scatter(x=dates, y=comp, mode="lines", name=name))
        plots.append(_plot("decomposition", f"Decomposition (season length {m})", "model", 0.55, g,
                           "Trend, seasonal and residual components."))
    return plots


def explain_plots(importance: dict, lc: dict | None) -> list[dict]:
    plots = []
    imp = importance["importance"]
    top = list(imp.items())[:15][::-1]
    method = importance["method"].replace("_", " ")
    plots.append(_plot("feature_importance", f"Feature importance ({method})", "explain", 0.75,
                       go.Figure(go.Bar(x=[v for _, v in top], y=[k for k, _ in top], orientation="h")),
                       ("Mean absolute SHAP value per original column." if not importance.get("substituted")
                        else "Permutation importance (SHAP was not available; see the report).")))
    if lc and lc.get("applicable"):
        f = go.Figure(go.Scatter(x=lc["sizes"], y=lc["val"], mode="lines+markers", name="validation"))
        f.add_trace(go.Scatter(x=lc["sizes"], y=lc["train"], mode="lines+markers", name="train"))
        f.update_xaxes(title="training rows")
        f.update_yaxes(title=lc["metric"])
        plots.append(_plot("learning_curve", "Learning curve", "explain", 0.45, f,
                           "If validation is still rising at the right, more data would help."))
    return plots


def rank_plots(plots: list[dict], statement: str) -> list[dict]:
    st = statement.lower()
    for p in plots:
        if any(w in st for w in KEYWORD_BOOST.get(p["id"], [])):
            p["relevance"] = round(min(1.0, p["relevance"] + 0.3), 3)
    return sorted(plots, key=lambda p: -p["relevance"])
