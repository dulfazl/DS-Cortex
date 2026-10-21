"""Loading, profiling and data-quality tools. Every function returns JSON-ready dicts
with executed results; agents copy these into Evidence objects, never paraphrase them."""
from __future__ import annotations

import csv
import io
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.common import DataError, jsonable

TARGET_HINTS = ("target", "label", "class", "outcome", "churn", "default", "survived", "fraud",
                "price", "sales", "revenue", "score", "y", "result", "response", "converted")
ID_HINTS = ("id", "uuid", "key", "index", "code", "number", "no")


# --------------------------------------------------------------------------- loading
def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a CSV robustly: BOM, Latin-1, semicolon/tab delimiters, blank or duplicate headers."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"File not found: {path.name}")
    raw = path.read_bytes()[:65536]
    if not raw.strip():
        raise DataError("The file is empty.")
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:  # pragma: no cover - latin-1 decodes anything
        raise DataError("Could not decode the file. Save it as UTF-8 and try again.")
    sep = _sniff_delimiter(text)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(path, sep=sep, encoding=encoding, nrows=nrows, low_memory=False)
    except pd.errors.EmptyDataError as e:
        raise DataError("The file has no columns.") from e
    except pd.errors.ParserError as e:
        raise DataError(f"The CSV could not be parsed: {str(e).splitlines()[0]}") from e
    if df.shape[1] == 0:
        raise DataError("The file has no columns.")
    if df.shape[0] == 0:
        raise DataError("The file has a header but no data rows.")
    df.columns = _clean_columns(df.columns)
    return df


def _clean_columns(cols) -> list[str]:
    out, seen = [], {}
    for i, c in enumerate(cols):
        name = str(c).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{i}"
        name = re.sub(r"\.\d+$", "", name) if re.search(r"\.\d+$", name) and name.rsplit(".", 1)[0] in seen else name
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def describe_csv(path: str) -> dict:
    """Shape, dtypes and missing counts of a CSV file (the Phase 0 tool)."""
    df = load_csv(path)
    return jsonable({
        "rows": df.shape[0],
        "columns": df.shape[1],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in df.isna().sum().items()},
    })


# --------------------------------------------------------------------------- type inference
def _parse_ratio_numeric(s: pd.Series) -> float:
    vals = s.dropna().astype(str).str.strip().str.replace(",", "", regex=False)
    if vals.empty:
        return 0.0
    return float(pd.to_numeric(vals, errors="coerce").notna().mean())


def _parse_ratio_datetime(s: pd.Series) -> float:
    vals = s.dropna().astype(str).str.strip()
    if vals.empty:
        return 0.0
    if vals.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.5:
        return 0.0  # plain numbers are not dates
    sample = vals.sample(min(len(vals), 500), random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def infer_kind(s: pd.Series) -> dict:
    """Return the semantic kind of a column plus the facts behind the decision."""
    n = len(s)
    non_null = s.dropna()
    nunique = int(non_null.nunique())
    unique_ratio = nunique / max(1, len(non_null))
    info: dict[str, Any] = {"n_unique": nunique, "unique_ratio": round(unique_ratio, 4), "stored_as_text": None}
    name = str(s.name).lower()
    if len(non_null) == 0:
        return {**info, "kind": "empty"}
    if nunique == 1:
        return {**info, "kind": "constant"}
    if pd.api.types.is_bool_dtype(s):
        return {**info, "kind": "boolean"}
    if pd.api.types.is_datetime64_any_dtype(s):
        return {**info, "kind": "datetime"}
    if pd.api.types.is_numeric_dtype(s):
        is_int = pd.api.types.is_integer_dtype(s) or bool(np.all(np.mod(non_null, 1) == 0))
        if nunique == 2:
            return {**info, "kind": "boolean"}
        name_id = any(name == h or name.endswith("_" + h) or name.startswith(h + "_") for h in ID_HINTS)
        if is_int and unique_ratio > 0.98 and len(non_null) >= 20 and (name_id or non_null.is_monotonic_increasing):
            return {**info, "kind": "id_like"}
        return {**info, "kind": "numeric"}
    # text-like storage
    str_vals = non_null.astype(str).str.strip()
    lowered = set(str_vals.str.lower().unique()[:10])
    if nunique == 2 or (nunique <= 2 and lowered <= {"yes", "no", "true", "false", "y", "n", "0", "1"}):
        return {**info, "kind": "boolean"}
    if _parse_ratio_numeric(non_null) >= 0.95:
        return {**info, "kind": "numeric", "stored_as_text": "numeric"}
    if _parse_ratio_datetime(non_null) >= 0.9:
        return {**info, "kind": "datetime", "stored_as_text": "datetime"}
    avg_len = float(str_vals.str.len().mean())
    avg_words = float(str_vals.str.split().str.len().mean())
    if unique_ratio > 0.95 and len(non_null) >= 20 and avg_words < 3:
        return {**info, "kind": "id_like"}
    if avg_len > 40 and avg_words > 5:
        return {**info, "kind": "text"}
    return {**info, "kind": "categorical"}


def type_summary(df: pd.DataFrame) -> dict[str, dict]:
    return {c: infer_kind(df[c]) for c in df.columns}


def coerce_types(df: pd.DataFrame, kinds: dict[str, dict] | None = None) -> pd.DataFrame:
    """Convert numerics/dates stored as text into real dtypes (stateless, safe before splitting)."""
    kinds = kinds or type_summary(df)
    out = df.copy()
    for c, k in kinds.items():
        if c not in out:
            continue
        if k.get("stored_as_text") == "numeric":
            out[c] = pd.to_numeric(out[c].astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce")
        elif k.get("stored_as_text") == "datetime" or (k["kind"] == "datetime" and not pd.api.types.is_datetime64_any_dtype(out[c])):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[c] = pd.to_datetime(out[c], errors="coerce", format="mixed")
    return out


# --------------------------------------------------------------------------- quick profile
def quick_profile(path: str | Path, max_rows: int = 200_000) -> dict:
    """Fast first look: counts, inferred kinds, missing share, cardinality, candidate targets."""
    path = Path(path)
    df = load_csv(path)
    n_rows = len(df)
    sampled = n_rows > max_rows
    if sampled:
        df = df.sample(max_rows, random_state=0)
    kinds = type_summary(df)
    cols = []
    for c in df.columns:
        k = kinds[c]
        ex = [str(v)[:40] for v in df[c].dropna().unique()[:3]]
        cols.append({
            "name": c, "kind": k["kind"], "dtype": str(df[c].dtype),
            "missing_pct": round(100 * float(df[c].isna().mean()), 2),
            "n_unique": k["n_unique"], "unique_ratio": k["unique_ratio"],
            "stored_as_text": k.get("stored_as_text"), "examples": ex,
        })
    return jsonable({
        "file": path.name,
        "file_size_mb": round(path.stat().st_size / 1e6, 3),
        "rows": n_rows,
        "columns": df.shape[1],
        "sampled": sampled,
        "sample_rows": len(df),
        "column_info": cols,
        "candidate_targets": candidate_targets(cols),
        "datetime_columns": [c["name"] for c in cols if c["kind"] == "datetime"],
        "missing_pct_total": round(100 * float(df.isna().mean().mean()), 2),
    })


def candidate_targets(cols: list[dict]) -> list[str]:
    scored = []
    for i, c in enumerate(cols):
        if c["kind"] in ("id_like", "constant", "empty", "text", "datetime"):
            continue
        name = c["name"].lower()
        score = 0.0
        if any(name == h or name.startswith(h + "_") or name.endswith("_" + h) or name.startswith("is_") for h in TARGET_HINTS):
            score += 2
        if c["kind"] == "boolean":
            score += 1
        if i == len(cols) - 1:
            score += 1
        if score > 0:
            scored.append((score, c["name"]))
    return [n for _, n in sorted(scored, key=lambda t: -t[0])][:5]


def read_text_preview(path: str | Path, n: int = 5) -> str:
    df = load_csv(path, nrows=n)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()
