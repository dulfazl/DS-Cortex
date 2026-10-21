"""Small helpers shared by every tool: JSON-safe conversion and a readable error type."""
from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import numpy as np
import pandas as pd


class DataError(ValueError):
    """Raised for problems with the user's data. The message is shown to the user as is."""


def now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def jsonable(obj: Any, digits: int | None = 6) -> Any:
    """Convert numpy/pandas objects to plain JSON types. NaN and inf become None."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, digits) if digits is not None else f
    if isinstance(obj, dict):
        return {str(k): jsonable(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v, digits) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v, digits) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [jsonable(v, digits) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return jsonable(obj.model_dump(), digits)
    return str(obj)


def clip01(x: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return float(min(1.0, max(0.0, x)))
