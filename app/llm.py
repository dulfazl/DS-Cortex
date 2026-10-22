"""The only door to the language model. Gemini Flash plans, critiques and narrates; it never
computes. Without a key the system runs fully offline on deterministic templates.

Every narration passes a number check: if the LLM writes a number that no tool produced,
the text is rejected, counted, and replaced by the template."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app import db
from app.config import get_settings
from tools.verify_tools import numbers_supported

NARRATOR_SYSTEM = (
    "You are a careful data-science narrator inside DS Cortex. Use only the facts given in JSON. "
    "Never introduce a number that is not in the facts. Never claim certainty the facts do not support. "
    "Round numbers for readers (about three significant figures; percentages with one decimal). "
    "Write plain, short sentences.")


COOL_DOWN_SECONDS = 300          # an overloaded model is skipped for this long
MAX_SHORT_WAIT = 30              # when every model is briefly rate limited, wait at most this long, once

# Shared by every run in the process, so a new run does not re-try a model that is retired or out of quota.
_DEAD: set[str] = set()
_COOLING: dict[str, float] = {}
_state_lock = threading.Lock()


def _retry_delay(text: str) -> float | None:
    m = re.search(r"retryDelay'?:\s*'?(\d+(?:\.\d+)?)s", text)
    return float(m.group(1)) if m else None


class BudgetExceeded(RuntimeError):
    pass


class BaseLLM:
    name = "offline"

    def __init__(self) -> None:
        self.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "failures": 0,
                      "rejected_narrations": 0, "llm_narrations": 0}

    @property
    def available(self) -> bool:
        return False

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False,
                 temperature: float = 0.2) -> str | None:
        return None


class OfflineLLM(BaseLLM):
    name = "offline"


class GeminiLLM(BaseLLM):
    """Gemini through a model chain. A retired model (404) is dropped; an overloaded one (503) or one out
    of its daily quota (429 with a long retry delay) is skipped for a while and the next model answers;
    a short per-minute limit is waited out on the same model. When every model is unavailable the LLM
    reports itself unavailable and the run uses template narration instead of waiting."""
    name = "gemini"

    def __init__(self, api_key: str, model: str, fallbacks: tuple[str, ...] = (), max_retries: int = 4):
        super().__init__()
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.fallbacks = tuple(m for m in fallbacks if m and m != model)
        self.max_retries = max_retries
        self.name = f"gemini:{model}"
        self.usage["models"] = {}

    def _chain(self) -> list[str]:
        now = time.monotonic()
        with _state_lock:
            return [m for m in (self.model, *self.fallbacks) if m not in _DEAD and _COOLING.get(m, 0) <= now]

    @property
    def available(self) -> bool:
        """A model is usable now, or within a short per-minute wait that generate() sits out."""
        if self._chain():
            return True
        soonest = self._soonest_free()
        return soonest is not None and soonest <= MAX_SHORT_WAIT

    def _check_budget(self) -> None:
        budget = get_settings().daily_call_budget
        day = dt.date.today().isoformat()
        with db.connect() as c:
            row = c.execute("SELECT calls FROM usage WHERE day=?", (day,)).fetchone()
        if row and row["calls"] >= budget:
            raise BudgetExceeded(f"Daily Gemini call budget of {budget} reached; try again tomorrow.")

    def _count(self, tokens: int) -> None:
        day = dt.date.today().isoformat()
        with db.connect(write=True) as c:
            c.execute("INSERT INTO usage(day, calls, tokens) VALUES (?,1,?) ON CONFLICT(day) DO UPDATE "
                      "SET calls=calls+1, tokens=tokens+excluded.tokens", (day, tokens))

    def _soonest_free(self) -> float | None:
        """Seconds until the first cooling model of this chain is usable again (None if none is cooling)."""
        now = time.monotonic()
        with _state_lock:
            waits = [_COOLING[m] - now for m in (self.model, *self.fallbacks) if m not in _DEAD and _COOLING.get(m, 0) > now]
        return max(0.0, min(waits)) if waits else None

    @staticmethod
    def _cool(model: str, seconds: float) -> None:
        with _state_lock:
            _COOLING[model] = time.monotonic() + seconds

    def generate(self, prompt, system=None, json_mode=False, temperature=0.2):
        self._check_budget()
        types = self._genai.types
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                          response_mime_type="application/json" if json_mode else None)
        for _pass in range(2):   # a second pass only when every model is briefly rate limited
            for model in self._chain():
                for attempt in range(self.max_retries):
                    try:
                        resp = self.client.models.generate_content(model=model, contents=prompt, config=cfg)
                        um = getattr(resp, "usage_metadata", None)
                        pt = int(getattr(um, "prompt_token_count", 0) or 0)
                        ot = int(getattr(um, "candidates_token_count", 0) or 0)
                        self.usage["calls"] += 1
                        self.usage["prompt_tokens"] += pt
                        self.usage["output_tokens"] += ot
                        self.usage["models"][model] = self.usage["models"].get(model, 0) + 1
                        self._count(pt + ot)
                        return resp.text
                    except Exception as e:  # noqa: BLE001 - SDK raises several types
                        code = getattr(e, "code", None) or getattr(e, "status_code", None)
                        text = str(e)
                        if code == 404 or "NOT_FOUND" in text:
                            with _state_lock:
                                _DEAD.add(model)
                            break
                        if code == 429 or "RESOURCE_EXHAUSTED" in text:
                            # daily quota or per-minute limit: set the model aside for as long as Google asks
                            # and use the next one now, instead of sleeping while other models are free
                            self._cool(model, _retry_delay(text) or 10.0)
                            break
                        if code in (500, 503) or "UNAVAILABLE" in text:
                            if attempt >= 1:
                                self._cool(model, COOL_DOWN_SECONDS)
                                break
                            time.sleep(2.0)
                            continue
                        break
            soonest = self._soonest_free()
            if _pass or self._chain() or soonest is None or soonest > MAX_SHORT_WAIT:
                break
            time.sleep(soonest)       # every model is briefly limited: wait for the first one, once
        self.usage["failures"] += 1
        return None


class FakeLLM(BaseLLM):
    """Test double. Replies come from a function or from recorded cassettes keyed by prompt hash."""
    name = "fake"

    def __init__(self, reply: Callable[[str, str | None], str | None] | dict[str, str] | None = None,
                 cassette: Path | None = None, record_from: BaseLLM | None = None):
        super().__init__()
        self.reply = reply
        self.cassette = cassette
        self.record_from = record_from
        self.tape: dict[str, str] = json.loads(cassette.read_text()) if cassette and cassette.exists() else {}

    @property
    def available(self) -> bool:
        return True

    @staticmethod
    def key(prompt: str, system: str | None) -> str:
        return hashlib.sha256(((system or "") + "\n" + prompt).encode()).hexdigest()[:16]

    def generate(self, prompt, system=None, json_mode=False, temperature=0.2):
        self.usage["calls"] += 1
        k = self.key(prompt, system)
        if callable(self.reply):
            return self.reply(prompt, system)
        if isinstance(self.reply, dict) and k in self.reply:
            return self.reply[k]
        if k in self.tape:
            return self.tape[k]
        if self.record_from is not None:
            out = self.record_from.generate(prompt, system, json_mode, temperature)
            if out is not None and self.cassette:
                self.tape[k] = out
                self.cassette.write_text(json.dumps(self.tape, indent=1))
            return out
        raise KeyError(f"No recorded reply for prompt hash {k}: re-record cassettes")


def get_llm(mode: str | None = None) -> BaseLLM:
    s = get_settings()
    mode = mode or s.llm_mode
    if mode == "offline" or (mode == "auto" and not s.google_api_key):
        return OfflineLLM()
    if not s.google_api_key:
        raise RuntimeError("LLM_MODE=gemini but GOOGLE_API_KEY is not set")
    return GeminiLLM(s.google_api_key, s.gemini_model, s.gemini_fallbacks)


# --------------------------------------------------------------------------- helpers
def narrate(llm: BaseLLM, task: str, facts: dict[str, Any], fallback: str,
            depth: str = "technical") -> tuple[str, str]:
    """Return (text, source). The LLM may phrase; every number must trace to `facts`."""
    if not llm.available:
        return fallback, "template"
    style = ("Explain for a non-specialist in 3 to 5 sentences, no jargon." if depth == "beginner"
             else "Explain for a data scientist in 3 to 6 sentences.")
    prompt = (f"Task: {task}\n{style}\nFacts (JSON; the ONLY numbers you may use):\n"
              f"{json.dumps(facts, default=str)[:12000]}\n")
    text = llm.generate(prompt, system=NARRATOR_SYSTEM)
    if not text or not text.strip():
        return fallback, "template"
    ok, unmatched = numbers_supported(text, facts)
    if not ok:
        llm.usage["rejected_narrations"] += 1
        return fallback, f"template (LLM text rejected: unsupported numbers {unmatched[:5]})"
    llm.usage["llm_narrations"] += 1
    return text.strip(), "llm"


def ask_json(llm: BaseLLM, prompt: str, system: str | None = None) -> Any | None:
    if not llm.available:
        return None
    text = llm.generate(prompt, system=system, json_mode=True)
    if not text:
        return None
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_prompt(name: str) -> str:
    p = Path(__file__).resolve().parent.parent / "prompts" / f"{name}.md"
    return p.read_text() if p.exists() else ""
