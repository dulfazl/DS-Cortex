"""Evidence verification. A claim counts only if it cites a run_id that exists in the tool log
and every check holds against the *logged* result (not the copy inside the claim).
Also: number tracing, used to reject LLM narration that introduces numbers no tool produced."""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from app.schemas import Check, Claim
from tools.tool_log import ToolLog, lookup

NUM_RE = re.compile(r"(?<![A-Za-z0-9_.\-])[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![A-Za-z0-9_.\-])[-−]?\d+(?:\.\d+)?%?(?![A-Za-z0-9_])")
DEFAULT_ALLOWED = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0}


# --------------------------------------------------------------------------- claims
def _check(check: Check, result: dict) -> tuple[bool, str]:
    try:
        actual = lookup(result, check.path)
    except (KeyError, IndexError, ValueError):
        return False, f"path '{check.path}' not in result"
    op, v = check.op, check.value
    try:
        if op == "true":
            return bool(actual), ""
        if op == "false":
            return not bool(actual), ""
        if op == "contains":
            return v in actual, ""
        if op == "eq":
            return actual == v, ""
        a, b = float(actual), float(v)
        if op == "approx":
            ok = math.isclose(a, b, rel_tol=check.tol, abs_tol=check.tol)
        elif op == "gt":
            ok = a > b
        elif op == "ge":
            ok = a >= b
        elif op == "lt":
            ok = a < b
        elif op == "le":
            ok = a <= b
        else:
            return False, f"unknown op {op}"
        return ok, "" if ok else f"{check.path}={actual} fails {op} {v}"
    except (TypeError, ValueError):
        return False, f"cannot compare {actual!r} with {v!r}"


def verify_claim(claim: Claim, log: ToolLog) -> Claim:
    if not claim.evidence or not claim.checks:
        claim.verified, claim.verify_note = False, "no evidence or no checkable support"
        return claim
    cited = {e.run_id for e in claim.evidence}
    for ev in claim.evidence:
        if log.get(ev.run_id) is None:
            claim.verified, claim.verify_note = False, f"run {ev.run_id[:8]} not in tool log"
            return claim
    for chk in claim.checks:
        if chk.run_id not in cited:
            claim.verified, claim.verify_note = False, f"check cites run {chk.run_id[:8]} that is not in its evidence"
            return claim
        rec = log.get(chk.run_id)
        ok, why = _check(chk, rec["result"])
        if not ok:
            claim.verified, claim.verify_note = False, why
            return claim
    claim.verified, claim.verify_note = True, None
    return claim


def verify_claims(claims: Iterable[Claim], log: ToolLog) -> dict[str, Any]:
    claims = list(claims)
    for c in claims:
        verify_claim(c, log)
    rejected = [c for c in claims if not c.verified]
    return {"total": len(claims), "verified": len(claims) - len(rejected), "rejected": len(rejected),
            "rejected_llm": sum(1 for c in rejected if c.source == "llm"),
            "total_llm": sum(1 for c in claims if c.source == "llm"),
            "rejected_texts": [f"{c.text} ({c.verify_note})" for c in rejected][:20]}


# --------------------------------------------------------------------------- number tracing
def _to_float(tok: str) -> tuple[float, int, bool]:
    pct = tok.endswith("%")
    t = tok.rstrip("%").replace(",", "").replace("−", "-")
    decimals = len(t.split(".")[1]) if "." in t else 0
    return float(t), decimals, pct


def extract_numbers(text: str) -> list[str]:
    return NUM_RE.findall(text or "")


def flatten_numbers(obj: Any, out: set[float] | None = None) -> set[float]:
    out = set() if out is None else out
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        if not (isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj))):
            out.add(float(obj))
    elif isinstance(obj, str):
        for tok in extract_numbers(obj):
            out.add(_to_float(tok)[0])
    elif isinstance(obj, dict):
        for v in obj.values():
            flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            flatten_numbers(v, out)
    return out


def number_supported(tok: str, facts: set[float], allowed: set[float] = DEFAULT_ALLOWED) -> bool:
    x, decimals, pct = _to_float(tok)
    if abs(x) in allowed:
        return True
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    for f in facts:
        cands = (f, f * 100) if pct or decimals <= 1 else (f,)
        for c in cands:
            if abs(abs(x) - abs(c)) <= max(tol, 1e-9 * abs(c)):
                return True
    return False


def numbers_supported(text: str, facts: Any, allowed: set[float] = DEFAULT_ALLOWED) -> tuple[bool, list[str]]:
    fact_nums = flatten_numbers(facts)
    unmatched = [t for t in extract_numbers(text) if not number_supported(t, fact_nums, allowed)]
    return (not unmatched), unmatched
