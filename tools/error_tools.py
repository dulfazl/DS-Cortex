"""Error analysis: confusion matrix and per-class metrics, residuals and worst predictions,
learning curves, cluster profiles, anomaly score summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools import pipeline_core as pc
from tools.common import jsonable


def error_analysis(task: dict, predictions: dict, X_test: pd.DataFrame | None = None) -> dict:
    fam = task["task_family"]
    out: dict = {"family": fam}
    if fam == "classification":
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support, roc_curve
        y, yp = np.asarray(predictions["y_true"]), np.asarray(predictions["y_pred"])
        classes = predictions["classes"]
        k = len(classes)
        cm = confusion_matrix(y, yp, labels=list(range(k)))
        p, r, f, sup = precision_recall_fscore_support(y, yp, labels=list(range(k)), zero_division=0)
        out.update(confusion_matrix=cm.tolist(), classes=classes,
                   per_class={classes[i]: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(sup[i])} for i in range(k)})
        proba = predictions.get("proba")
        if proba is not None and k == 2:
            pos = predictions.get("pos") or 1
            s = np.asarray(proba)[:, pos]
            yb = (y == pos).astype(int)
            fpr, tpr, _ = roc_curve(yb, s)
            pr, rc, _ = precision_recall_curve(yb, s)
            frac, mean_pred = calibration_curve(yb, s, n_bins=min(10, max(2, len(y) // 20)), strategy="quantile")
            out.update(roc={"fpr": fpr.tolist(), "tpr": tpr.tolist()}, pr={"precision": pr.tolist(), "recall": rc.tolist()},
                       calibration={"mean_predicted": mean_pred.tolist(), "fraction_positive": frac.tolist()},
                       positive_class=classes[pos])
    elif fam == "regression":
        y, yp = np.asarray(predictions["y_true"], float), np.asarray(predictions["y_pred"], float)
        res = y - yp
        worst = np.argsort(-np.abs(res))[:10]
        out.update(residual_mean=float(res.mean()), residual_std=float(res.std()), y_true=y.tolist(), y_pred=yp.tolist(),
                   worst=[{"row": int(i), "y_true": float(y[i]), "y_pred": float(yp[i]), "abs_error": float(abs(res[i]))} for i in worst])
    elif fam == "clustering":
        labels = np.asarray(predictions["labels"])
        out["sizes"] = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    elif fam == "anomaly_detection":
        s = np.asarray(predictions["scores"], float)
        thr = float(np.quantile(s, 0.95))
        out.update(threshold=thr, flagged=int((s > thr).sum()), scores=s.tolist(),
                   top_rows=[int(i) for i in np.argsort(-s)[:10]])
    elif fam == "time_series":
        out.update({k: predictions[k] for k in ("dates", "y", "forecast", "half_width", "n_test", "season")})
    return jsonable(out)


def learning_curve(spec: dict, df_train: pd.DataFrame, task: dict, seed: int, sizes=(0.3, 0.6, 1.0)) -> dict:
    if task["task_family"] not in ("classification", "regression"):
        return {"applicable": False}
    info = pc.prepare_supervised(df_train, task)
    rng = np.random.default_rng(seed)
    val, train = [], []
    for frac in sizes:
        v, t = [], []
        for tr, te in pc.splitter(task, len(info["y"]), 3, 1, seed, info["y"], info["groups"]):
            sub = rng.choice(tr, max(10, int(len(tr) * frac)), replace=False) if frac < 1 else tr
            res, _, _ = pc._fit_predict_supervised(spec, task, info, sub, te, seed, [task["metric"]])
            v.append(res["val"][task["metric"]])
            t.append(res["train"][task["metric"]])
        val.append(float(np.nanmean(v)))
        train.append(float(np.nanmean(t)))
    n = len(info["y"]) * 2 // 3
    return jsonable({"applicable": True, "sizes": [int(n * f) for f in sizes], "val": val, "train": train,
                     "metric": task["metric"]})


def cluster_profiles(df: pd.DataFrame, labels: list[int], task: dict, seed: int) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_samples
    X, cols = pc.unsupervised_matrix(df, task, seed)
    lab = np.asarray(labels)
    num = [c for c in cols["numeric"]]
    prof = df.assign(_c=lab).groupby("_c")[num].mean() if num else pd.DataFrame()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(len(X), 2000), replace=False)
    coords = PCA(n_components=2, random_state=seed).fit_transform(X)[idx] if X.shape[1] >= 2 else np.c_[X[idx, 0], np.zeros(len(idx))]
    sil = silhouette_samples(X[idx], lab[idx]).tolist() if len(np.unique(lab[idx][lab[idx] >= 0])) > 1 else []
    return jsonable({"profiles": {str(k): row.to_dict() for k, row in prof.iterrows()}, "features": num,
                     "pca": {"x": coords[:, 0].tolist(), "y": coords[:, 1].tolist(), "label": lab[idx].tolist()},
                     "silhouette_samples": sil, "silhouette_labels": lab[idx].tolist()})
