"""The Gemini model chain, with a stub client: retired models are dropped, overloaded or out-of-quota
models are skipped for a while, short per-minute limits are waited out on the same model."""
from types import SimpleNamespace

import pytest

import app.llm as L
from app.llm import BaseLLM, GeminiLLM


class ApiError(Exception):
    def __init__(self, code, text=""):
        super().__init__(f"{code} {text}")
        self.code = code


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    L._DEAD.clear()
    L._COOLING.clear()
    monkeypatch.setattr("app.llm.time.sleep", lambda s: None)
    yield
    L._DEAD.clear()
    L._COOLING.clear()


def _llm(behaviour):
    calls = []

    def generate_content(model, contents, config):
        calls.append(model)
        step = behaviour[model].pop(0) if behaviour[model] else "ok"
        if step != "ok":
            raise ApiError(*step) if isinstance(step, tuple) else ApiError(step)
        return SimpleNamespace(text=f"from {model}", usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=1))

    llm = GeminiLLM.__new__(GeminiLLM)
    BaseLLM.__init__(llm)
    from google import genai
    llm._genai = genai
    llm.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    llm.model, llm.fallbacks, llm.max_retries, llm.name = "primary", ("second", "third"), 4, "gemini:primary"
    llm.usage["models"] = {}
    return llm, calls


DAILY = (429, "RESOURCE_EXHAUSTED {'retryDelay': '8540s'}")
MINUTE = (429, "RESOURCE_EXHAUSTED {'retryDelay': '7s'}")


def test_retired_model_is_dropped(monkeypatch):
    llm, calls = _llm({"primary": [404], "second": [], "third": []})
    assert llm.generate("hi") == "from second"
    llm.generate("again")
    assert calls == ["primary", "second", "second"]


def test_overloaded_model_retries_once_then_cools_down(monkeypatch):
    llm, calls = _llm({"primary": [503, 503], "second": [], "third": []})
    assert llm.generate("hi") == "from second"
    llm.generate("again")
    assert calls == ["primary", "primary", "second", "second"]


def test_daily_quota_switches_model_and_is_remembered_by_new_runs(monkeypatch):
    llm, calls = _llm({"primary": [DAILY], "second": [], "third": []})
    assert llm.generate("hi") == "from second"
    fresh, fresh_calls = _llm({"primary": [], "second": [], "third": []})
    assert fresh.generate("hi") == "from second" and fresh_calls == ["second"]


def test_per_minute_limit_waits_on_the_same_model(monkeypatch):
    llm, calls = _llm({"primary": [MINUTE, MINUTE], "second": [], "third": []})
    assert llm.generate("hi") == "from primary"
    assert calls == ["primary"] * 3


def test_all_models_out_means_unavailable(monkeypatch):
    llm, calls = _llm({"primary": [DAILY], "second": [DAILY], "third": [DAILY]})
    assert llm.generate("hi") is None and llm.usage["failures"] == 1
    assert llm.available is False          # the run now uses templates without waiting
