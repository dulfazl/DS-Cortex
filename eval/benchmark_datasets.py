"""The benchmark suite: 9 public datasets, downloaded once and cached, so every run uses the same
data. Each entry carries the problem statement and the expected task plan (used only to answer the
router's clarifying question non-interactively, never to set the metric)."""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from app.config import ROOT

CACHE = ROOT / "datasets" / "benchmark"
MAX_ROWS = 3000   # pre-registered cap so the full ablation stays within a day on a laptop


@dataclass
class Benchmark:
    name: str
    family: str
    statement: str
    target: str | None
    loader: Callable[[], pd.DataFrame]
    source: str
    licence: str
    hidden_label: str | None = None          # clustering: a label removed from the data, kept for reference only
    notes: list[str] = field(default_factory=list)


def _download(url: str, name: str) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:   # noqa: S310 - fixed, public URLs
            path.write_bytes(r.read())
    return path.read_bytes()


def _csv(url: str, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_download(url, name)), **kw)


GH = "https://raw.githubusercontent.com"


def _titanic():
    return _csv(f"{GH}/datasciencedojo/datasets/master/titanic.csv", "titanic.csv")


def _telco():
    return _csv(f"{GH}/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv", "telco_churn.csv")


def _oil():
    df = _csv(f"{GH}/jbrownlee/Datasets/master/oil-spill.csv", "oil_spill.csv", header=None)
    df.columns = [f"f{i}" for i in range(df.shape[1] - 1)] + ["spill"]
    return df


def _housing():
    return _csv(f"{GH}/ageron/handson-ml/master/datasets/housing/housing.csv", "california_housing.csv")


def _diabetes():
    from sklearn.datasets import load_diabetes
    d = load_diabetes(as_frame=True, scaled=False)
    return d.frame.rename(columns={"target": "progression"})


def _penguins():
    return _csv(f"{GH}/mwaskom/seaborn-data/master/penguins.csv", "penguins.csv")


def _wine():
    from sklearn.datasets import load_wine
    return load_wine(as_frame=True).frame.rename(columns={"target": "cultivar"})


def _airline():
    return _csv(f"{GH}/jbrownlee/Datasets/master/airline-passengers.csv", "airline_passengers.csv")


def _mammography():
    df = _csv(f"{GH}/jbrownlee/Datasets/master/mammography.csv", "mammography.csv", header=None)
    df.columns = [f"f{i}" for i in range(df.shape[1] - 1)] + ["is_anomaly"]
    df["is_anomaly"] = (df["is_anomaly"].astype(str).str.strip("'\" ") == "1").astype(int)
    return df


SUITE: list[Benchmark] = [
    Benchmark("titanic", "classification", "Predict which passengers Survived", "Survived", _titanic,
              "datasciencedojo/datasets (Kaggle Titanic)", "public domain (Kaggle competition data)"),
    Benchmark("telco_churn", "classification", "Predict customer Churn", "Churn", _telco,
              "IBM telco-customer-churn-on-icp4d", "Apache-2.0"),
    Benchmark("oil_spill", "classification", "Predict whether each image shows an oil spill; the label column is spill",
              "spill", _oil, "jbrownlee/Datasets (Kubat et al., 1998)", "research use", notes=["imbalanced"]),
    Benchmark("california_housing", "regression", "Estimate median_house_value for each district", "median_house_value",
              _housing, "ageron/handson-ml (1990 US census)", "public domain"),
    Benchmark("diabetes", "regression", "Estimate the disease progression one year later", "progression", _diabetes,
              "scikit-learn (Efron et al., 2004)", "BSD (scikit-learn)"),
    Benchmark("penguins", "clustering", "Segment the penguins into natural groups", None, _penguins,
              "mwaskom/seaborn-data (Palmer Penguins)", "CC0", hidden_label="species"),
    Benchmark("wine", "clustering", "Segment these wines into natural groups", None, _wine,
              "scikit-learn (UCI Wine)", "CC BY 4.0", hidden_label="cultivar"),
    Benchmark("airline_passengers", "time_series", "Forecast monthly Passengers for the next months", "Passengers",
              _airline, "jbrownlee/Datasets (Box and Jenkins)", "public domain"),
    Benchmark("mammography", "anomaly_detection",
              "Find unusual mammography readings; the label column is_anomaly marks known anomalies", "is_anomaly",
              _mammography, "jbrownlee/Datasets (Woods et al., 1993)", "research use"),
]


def get(name: str) -> Benchmark:
    return next(b for b in SUITE if b.name == name)


def load(b: Benchmark, seed: int = 0, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    df = b.loader()
    if b.hidden_label and b.hidden_label in df:
        df = df.drop(columns=[b.hidden_label])
    if b.family != "time_series" and len(df) > max_rows:
        strat = df[b.target] if b.family in ("classification", "anomaly_detection") else None
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=max_rows, random_state=0, stratify=strat)
    return df.reset_index(drop=True)


def materialize(b: Benchmark, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{b.name}.csv"
    if not path.exists():
        load(b).to_csv(path, index=False)
    return path


def describe_suite() -> pd.DataFrame:
    rows = []
    for b in SUITE:
        try:
            df = load(b)
            rows.append({"dataset": b.name, "task": b.family, "rows": len(df), "features": df.shape[1] - (1 if b.target else 0),
                         "target": b.target or f"none (hidden: {b.hidden_label})", "missing_pct": round(100 * float(df.isna().mean().mean()), 2),
                         "source": b.source, "licence": b.licence})
        except Exception as e:  # noqa: BLE001 - an unreachable source is reported, not fatal
            rows.append({"dataset": b.name, "task": b.family, "error": f"{type(e).__name__}: {e}"})
    return pd.DataFrame(rows)
