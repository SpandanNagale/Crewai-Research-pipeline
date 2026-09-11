# LLM Gateway Fallback (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `get_llm(agent_name)` conditional fallback across 5 providers
(ollama_local, ollama_cloud, groq, openrouter, gemini) — timeout/5xx retries once then
advances, rate-limit/auth advances immediately, a failed deployment cools down for 600s,
a refusal is never retried or failed over — without changing what any agent module calls
or touching the tool-calling / structured-output paths that already work.

**Architecture:** `llm/provider.py` keeps its existing `LLMProvider` ABC and adds two more
concrete classes (OpenRouter, Gemini) alongside the three that already exist. A new
`llm/router.py` builds, per agent-name alias, an ordered list of real `crewai.LLM`
instances from `config/agents_config.yaml`'s new `model_groups:` section. A new
`llm/router_llm.py` defines `RouterLLM(BaseLLM)`, which holds that list and implements
the fallback/cooldown state machine in its `call()`, delegating each attempt to the real
`crewai.LLM.call()` of the currently-selected deployment (so tool-calling, `num_ctx`, and
structured-output parsing are untouched — they're crewai's own code, unmodified).
`get_llm(agent_name)` now returns a `RouterLLM` instead of a single provider's `LLM`.

**Tech Stack:** Python, crewai 1.14.2 (`crewai.llm.BaseLLM`, `crewai.LLM`), openai SDK
exception types (raised natively by crewai's per-provider client — no litellm involved),
pytest 9.1.1, FastAPI (for the new health endpoint).

**Spec:** `docs/superpowers/specs/2026-09-11-litellm-router-design.md`

## Global Constraints

- No litellm dependency — confirmed in the spec's revision note that routing real calls
  through `litellm.Router.completion()` would bypass crewai's tool-execution loop, the
  Ollama `num_ctx` fix, and structured-output parsing.
- `get_llm(agent_name)`'s signature and import path (`llm.provider.get_llm`) must not
  change — `agents/base.py` and `eval/scorer.py` import it directly.
- Every agent-name alias already in use — `planner`, `researcher`, `evidence_comparator`,
  `classifier`, `writer`, `critic`, `evaluator` — keeps its exact string; no renaming.
- Retry policy is fixed: timeout/5xx → 1 retry same deployment after 1s backoff, then
  advance; rate-limit/auth/not-found → 0 retries, advance immediately; cooldown 600s;
  refusal → never retried or failed over.
- `crew/research_crew.py` requires zero changes — verified that crewai's native
  per-provider client raises plain `openai.*` exceptions directly (not litellm's), so
  `_is_provider_unavailable_error`'s existing `isinstance` check already covers every
  exception `RouterLLM.call()` can raise.
- Model names placed into `config/agents_config.yaml`'s `model_groups` in Task 3 are
  reasonable placeholders, explicitly corrected against real `GET /health/providers`
  output in Task 6 — never guess-and-ship without that verification pass.

---

### Task 1: New provider classes (OpenRouter, Gemini) + settings + dev dependencies

**Files:**
- Modify: `config/settings.py`
- Modify: `.env.example`
- Modify: `llm/provider.py`
- Create: `requirements-dev.txt`
- Test: `tests/test_provider_configs.py` (new file — first test file in the repo, needs `tests/__init__.py`)
- Create: `tests/__init__.py` (empty)

**Interfaces:**
- Produces: `llm.provider.OpenRouterProvider`, `llm.provider.GeminiProvider` — both
  `LLMProvider` subclasses with `.build(model: str, temperature: float) -> LLM`, matching
  the existing `OllamaLocalProvider` / `OllamaCloudProvider` / `GroqProvider` shape.
  `llm.provider._PROVIDERS` dict extended with keys `"openrouter"` and `"gemini"`.
  `config.settings.settings.OPENROUTER_API_KEY`, `.OPENROUTER_BASE_URL`,
  `.GEMINI_API_KEY` (all `str`, default `""` for the two keys, sensible default URL for
  `OPENROUTER_BASE_URL`).

- [ ] **Step 1: Add the new settings fields**

Edit `config/settings.py`, inserting after the existing `GROQ_BASE_URL` line (after line 20):

```python
    # ── OpenRouter (opt-in) ──────────────────────────────────────────
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # ── Gemini (opt-in) ──────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
```

- [ ] **Step 2: Document the new keys in `.env.example`**

Append to `.env.example`, after the existing Groq block:

```
# ── OpenRouter (opt-in) ──────────────────────────────────────────────
# Aggregates many hosted models behind one API. Get a key at
# https://openrouter.ai/keys. Used as a fallback deployment for every agent.
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# ── Gemini (opt-in) ────────────────────────────────────────────────────
# Google's hosted models. Get a key at https://aistudio.google.com/apikey.
# Used as a fallback deployment for every agent.
GEMINI_API_KEY=
```

- [ ] **Step 3: Create `requirements-dev.txt`**

```
pytest==9.1.1
```

- [ ] **Step 4: Install the dev dependency**

Run: `venv/Scripts/python -m pip install -r requirements-dev.txt`

- [ ] **Step 5: Create the tests package**

Create `tests/__init__.py` (empty file).

- [ ] **Step 6: Write the failing tests**

Create `tests/test_provider_configs.py`:

```python
import pytest

from llm.provider import (
    GeminiProvider,
    LLMConfigError,
    OpenRouterProvider,
)


def test_openrouter_provider_raises_without_key(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.OPENROUTER_API_KEY", "")
    with pytest.raises(LLMConfigError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider().build("meta-llama/llama-3.3-70b-instruct:free", 0.3)


def test_openrouter_provider_builds_with_key(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "llm.provider.settings.OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    llm = OpenRouterProvider().build("meta-llama/llama-3.3-70b-instruct:free", 0.3)
    assert llm.model == "meta-llama/llama-3.3-70b-instruct:free"
    assert llm.provider == "openrouter"
    assert llm.api_key == "sk-or-test"


def test_gemini_provider_raises_without_key(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.GEMINI_API_KEY", "")
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        GeminiProvider().build("gemini-2.0-flash", 0.3)


def test_gemini_provider_builds_with_key(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.GEMINI_API_KEY", "test-gemini-key")
    llm = GeminiProvider().build("gemini-2.0-flash", 0.3)
    assert llm.model == "gemini-2.0-flash"
    assert llm.provider == "gemini"
    assert llm.api_key == "test-gemini-key"


def test_providers_dict_has_all_five():
    from llm.provider import _PROVIDERS

    assert set(_PROVIDERS.keys()) == {
        "ollama_local",
        "ollama_cloud",
        "groq",
        "openrouter",
        "gemini",
    }
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_provider_configs.py -v`
Expected: FAIL — `ImportError: cannot import name 'GeminiProvider'` (classes don't exist yet).

- [ ] **Step 8: Add the two provider classes to `llm/provider.py`**

Insert after the existing `GroqProvider` class (after the `class GroqProvider` block, before `_PROVIDERS = {`):

```python
class OpenRouterProvider(LLMProvider):
    """OpenRouter — aggregates many hosted models behind one OpenAI-compatible API."""

    def build(self, model: str, temperature: float) -> LLM:
        if not settings.OPENROUTER_API_KEY:
            raise LLMConfigError(
                "OPENROUTER_API_KEY is required to use the openrouter provider "
                "(get one at https://openrouter.ai/keys)."
            )
        return LLM(
            model=model,
            provider="openrouter",
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            temperature=temperature,
        )


class GeminiProvider(LLMProvider):
    """Google Gemini. crewai has a native client for this provider — no base_url needed."""

    def build(self, model: str, temperature: float) -> LLM:
        if not settings.GEMINI_API_KEY:
            raise LLMConfigError(
                "GEMINI_API_KEY is required to use the gemini provider "
                "(get one at https://aistudio.google.com/apikey)."
            )
        return LLM(
            model=model,
            provider="gemini",
            api_key=settings.GEMINI_API_KEY,
            temperature=temperature,
        )
```

Then update the `_PROVIDERS` dict:

```python
_PROVIDERS = {
    "ollama_local": OllamaLocalProvider(),
    "ollama_cloud": OllamaCloudProvider(),
    "groq": GroqProvider(),
    "openrouter": OpenRouterProvider(),
    "gemini": GeminiProvider(),
}
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_provider_configs.py -v`
Expected: PASS (5 tests)

- [ ] **Step 10: Commit**

```bash
git add config/settings.py .env.example llm/provider.py requirements-dev.txt tests/__init__.py tests/test_provider_configs.py
git commit -m "feat: add OpenRouter and Gemini LLM providers"
```

---

### Task 2: RouterLLM fallback/cooldown/refusal state machine

**Files:**
- Create: `llm/router_llm.py`
- Test: `tests/test_router_llm.py`

**Interfaces:**
- Consumes: nothing from Task 1 (this task is fully decoupled from the provider classes —
  it operates on any object with a `.call(...)` method matching `crewai.LLM.call`'s
  signature, tested against stub objects).
- Produces: `llm.router_llm.RouterLLM` (a `crewai.llm.BaseLLM` subclass with a
  `deployments: list` pydantic field), `llm.router_llm.GuardrailRefusalError`
  (`RuntimeError` subclass).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_router_llm.py`:

```python
import time

import openai
import pytest

from llm.router_llm import GuardrailRefusalError, RouterLLM


class FakeDeployment:
    """Stands in for a crewai.LLM instance in tests — no real network calls."""

    def __init__(self, name, script):
        self.name = name
        self._script = list(script)  # list of return values or exceptions to raise, in order
        self.calls = 0
        self.model = name
        self.temperature = 0.3
        self.timeout = None

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        self.calls += 1
        outcome = self._script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def get_context_window_size(self):
        return 8192

    def supports_stop_words(self):
        return True

    def supports_multimodal(self):
        return False

    def get_token_usage_summary(self):
        return {}


def _router(deployments):
    return RouterLLM(model="test-alias", deployments=deployments)


def test_first_deployment_succeeds_immediately():
    d1 = FakeDeployment("a", ["final answer"])
    router = _router([d1])
    assert router.call("hi") == "final answer"
    assert d1.calls == 1


def test_timeout_retries_same_deployment_once_then_advances():
    timeout_err = openai.APITimeoutError(request=None)
    d1 = FakeDeployment("a", [timeout_err, timeout_err])
    d2 = FakeDeployment("b", ["from b"])
    router = _router([d1, d2])
    result = router.call("hi")
    assert result == "from b"
    assert d1.calls == 2  # one retry on the same deployment
    assert d2.calls == 1


def test_rate_limit_advances_immediately_without_retry():
    rl_err = openai.RateLimitError(
        "rate limited", response=_fake_response(429), body=None
    )
    d1 = FakeDeployment("a", [rl_err])
    d2 = FakeDeployment("b", ["from b"])
    router = _router([d1, d2])
    result = router.call("hi")
    assert result == "from b"
    assert d1.calls == 1  # no retry on the same deployment


def test_failed_deployment_cools_down_across_calls():
    rl_err = openai.RateLimitError(
        "rate limited", response=_fake_response(429), body=None
    )
    d1 = FakeDeployment("a", [rl_err, "a is back"])
    d2 = FakeDeployment("b", ["from b", "from b again"])
    router = _router([d1, d2])

    first = router.call("hi")
    assert first == "from b"

    second = router.call("hi again")
    assert second == "from b again"
    assert d1.calls == 1  # d1 stayed in cooldown, never retried on the second call


def test_refusal_is_raised_uncaught_no_retry_no_fallback():
    d1 = FakeDeployment("a", ["I cannot assist with that request."])
    d2 = FakeDeployment("b", ["from b"])
    router = _router([d1, d2])
    with pytest.raises(GuardrailRefusalError):
        router.call("hi")
    assert d1.calls == 1
    assert d2.calls == 0  # never fell over to the next deployment


def test_all_deployments_exhausted_raises_last_error():
    auth_err = openai.AuthenticationError(
        "bad key", response=_fake_response(401), body=None
    )
    d1 = FakeDeployment("a", [auth_err])
    d2 = FakeDeployment("b", [auth_err])
    router = _router([d1, d2])
    with pytest.raises(openai.AuthenticationError):
        router.call("hi")


def _fake_response(status_code):
    import httpx

    request = httpx.Request("POST", "https://example.invalid")
    return httpx.Response(status_code, request=request)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_router_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm.router_llm'`

- [ ] **Step 3: Write `llm/router_llm.py`**

```python
"""
RouterLLM — conditional fallback across an ordered list of already-built crewai.LLM
deployments, for one agent-name alias.

Deliberately does NOT route calls through litellm: crewai.LLM.call() runs the
tool-execution loop (available_functions), the Ollama num_ctx fix, and this codebase's
raw-JSON structured-output parsing internally. Each attempt here calls a real
deployment's own .call() unchanged, so all of that keeps working — this class only
decides which deployment gets each attempt and when to give up on one.
"""
from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

import openai
from crewai.llm import BaseLLM
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

_COOLDOWN_SECONDS = 600.0
_TRANSIENT_RETRY_BACKOFF_SECONDS = 1.0

_TRANSIENT_EXCEPTIONS = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)
_IMMEDIATE_ADVANCE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
)

_REFUSAL_PHRASES = (
    "i cannot assist",
    "i can't help with that",
    "i'm not able to provide",
    "as an ai language model, i cannot",
    "i won't provide",
    "i refuse to",
)


class GuardrailRefusalError(RuntimeError):
    """
    Raised when a deployment's response looks like a refusal. Never caught by
    RouterLLM.call()'s except clauses, so it is never retried or failed over —
    laundering a refusal through a different vendor until one complies would be
    the opposite of a guardrail. This is a minimal placeholder check; the real
    guardrail taxonomy is a later phase.
    """


def _raise_if_refusal(text: str) -> None:
    lowered = text.strip().lower()
    for phrase in _REFUSAL_PHRASES:
        if lowered.startswith(phrase):
            raise GuardrailRefusalError(f"Deployment response looked like a refusal: {text[:200]!r}")


class RouterLLM(BaseLLM):
    """
    A BaseLLM that fans out to a list of real deployments in order, with
    conditional fallback. `model` is the agent-name alias, for identification/
    logging only — it is never sent to any provider.
    """

    deployments: List[Any]

    _cooldown_until: dict = PrivateAttr(default_factory=dict)

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        now = time.monotonic()
        order = [
            i for i in range(len(self.deployments))
            if self._cooldown_until.get(i, 0.0) <= now
        ]
        if not order:
            order = list(range(len(self.deployments)))

        last_error: Optional[Exception] = None
        for i in order:
            deployment = self.deployments[i]
            for attempt in range(2):
                try:
                    result = deployment.call(
                        messages, tools, callbacks, available_functions,
                        from_task, from_agent, response_model,
                    )
                except _TRANSIENT_EXCEPTIONS as e:
                    last_error = e
                    if attempt == 0:
                        logger.warning(
                            "Deployment %s transient error (attempt 1/2), retrying: %s",
                            getattr(deployment, "model", i), e,
                        )
                        time.sleep(_TRANSIENT_RETRY_BACKOFF_SECONDS)
                        continue
                    logger.warning(
                        "Deployment %s still failing after retry, advancing: %s",
                        getattr(deployment, "model", i), e,
                    )
                    self._cooldown_until[i] = time.monotonic() + _COOLDOWN_SECONDS
                    break
                except _IMMEDIATE_ADVANCE_EXCEPTIONS as e:
                    last_error = e
                    logger.warning(
                        "Deployment %s unavailable, advancing immediately: %s",
                        getattr(deployment, "model", i), e,
                    )
                    self._cooldown_until[i] = time.monotonic() + _COOLDOWN_SECONDS
                    break
                else:
                    if isinstance(result, str):
                        _raise_if_refusal(result)
                    return result

        if last_error is not None:
            raise last_error
        raise RuntimeError("RouterLLM: no deployments configured")

    def get_context_window_size(self) -> int:
        return self.deployments[0].get_context_window_size()

    def supports_stop_words(self) -> bool:
        return self.deployments[0].supports_stop_words()

    def supports_multimodal(self) -> bool:
        return self.deployments[0].supports_multimodal()

    def get_token_usage_summary(self):
        return self.deployments[0].get_token_usage_summary()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_router_llm.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add llm/router_llm.py tests/test_router_llm.py
git commit -m "feat: add RouterLLM fallback/cooldown/refusal state machine"
```

---

### Task 3: Deployment builder, model_groups config, and health check

**Files:**
- Modify: `config/agents_config.yaml`
- Create: `llm/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: `llm.provider._PROVIDERS` (Task 1), `llm.provider.LLMConfigError` (existing).
  Does NOT consume `llm.router_llm.RouterLLM` (Task 2) — this task only builds plain
  `crewai.LLM` deployment lists; wrapping them in a `RouterLLM` happens in Task 4.
- Produces: `llm.router.get_deployments(agent_name: str) -> list[LLM]` (cached per
  agent_name), `llm.router.check_provider_health() -> list[dict]` — each dict
  `{"agent_name": str, "provider": str, "model": str, "status": "ok" | "error", "detail": str}`.

- [ ] **Step 1: Replace the stale commented override block in `config/agents_config.yaml`**

Remove lines 65-80 (the commented-out `# Per-agent LLM overrides` block, from
`# Per-agent LLM overrides (all optional...` through the blank line before `evaluator:`)
and insert, right before the `evaluator:` block:

```yaml
# Routing: for each agent-name alias, an ordered list of {provider, model} deployments.
# RouterLLM (llm/router_llm.py) tries them in order with conditional fallback. Provider
# keys must be one of llm.provider._PROVIDERS: ollama_local, ollama_cloud, groq,
# openrouter, gemini. Model names below are initial placeholders — verify with
# GET /health/providers and correct any that don't resolve.
model_groups:
  planner:
    - {provider: ollama_cloud, model: "gpt-oss:120b-cloud"}
    - {provider: groq, model: "llama-3.3-70b-versatile"}
    - {provider: openrouter, model: "meta-llama/llama-3.3-70b-instruct:free"}
    - {provider: gemini, model: "gemini-2.0-flash"}
  researcher:
    - {provider: groq, model: "llama-3.3-70b-versatile"}
    - {provider: ollama_cloud, model: "gpt-oss:120b-cloud"}
    - {provider: openrouter, model: "meta-llama/llama-3.3-70b-instruct:free"}
    - {provider: gemini, model: "gemini-2.0-flash"}
  evidence_comparator:
    - {provider: ollama_cloud, model: "gpt-oss:120b-cloud"}
    - {provider: groq, model: "llama-3.3-70b-versatile"}
    - {provider: openrouter, model: "meta-llama/llama-3.3-70b-instruct:free"}
    - {provider: gemini, model: "gemini-2.0-flash"}
  classifier:
    - {provider: ollama_cloud, model: "gpt-oss:120b-cloud"}
    - {provider: groq, model: "llama-3.3-70b-versatile"}
    - {provider: openrouter, model: "meta-llama/llama-3.3-70b-instruct:free"}
    - {provider: gemini, model: "gemini-2.0-flash"}
  writer:
    - {provider: ollama_cloud, model: "gpt-oss:120b-cloud"}
    - {provider: openrouter, model: "meta-llama/llama-3.3-70b-instruct:free"}
    - {provider: gemini, model: "gemini-2.0-flash"}
    - {provider: groq, model: "llama-3.3-70b-versatile"}
  critic:
    - {provider: ollama_cloud, model: "gpt-oss:120b-cloud"}
    - {provider: openrouter, model: "meta-llama/llama-3.3-70b-instruct:free"}
    - {provider: gemini, model: "gemini-2.0-flash"}
    - {provider: groq, model: "llama-3.3-70b-versatile"}
  evaluator:
    - {provider: ollama_cloud, model: "gpt-oss:120b-cloud"}
    - {provider: groq, model: "llama-3.3-70b-versatile"}
    - {provider: openrouter, model: "meta-llama/llama-3.3-70b-instruct:free"}
    - {provider: gemini, model: "gemini-2.0-flash"}

```

(The `evaluator:` block with its `role`/`goal`/`backstory`/`llm_temperature: 0.0` keeps
following directly after, unchanged.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_router.py`:

```python
import pytest

from llm.provider import LLMConfigError
from llm.router_llm import RouterLLM


def test_get_deployments_returns_router_deployments_in_config_order(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.GROQ_API_KEY", "groq-test")
    monkeypatch.setattr("llm.provider.settings.OLLAMA_API_KEY", "ollama-test")
    monkeypatch.setattr("llm.provider.settings.OPENROUTER_API_KEY", "or-test")
    monkeypatch.setattr("llm.provider.settings.GEMINI_API_KEY", "gemini-test")

    import llm.router as router_module
    router_module._deployment_cache.clear()

    deployments = router_module.get_deployments("researcher")
    assert len(deployments) == 4
    assert deployments[0].provider == "openai"  # groq goes through crewai's openai-compatible client
    assert deployments[1].provider == "ollama"


def test_get_deployments_unknown_alias_raises():
    import llm.router as router_module
    router_module._deployment_cache.clear()

    with pytest.raises(LLMConfigError, match="no model_groups entry"):
        router_module.get_deployments("not_a_real_agent")


def test_get_deployments_missing_key_raises_actionable_error(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.GROQ_API_KEY", "")
    import llm.router as router_module
    router_module._deployment_cache.clear()

    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        router_module.get_deployments("researcher")


def test_check_provider_health_pings_every_deployment(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.GROQ_API_KEY", "groq-test")
    monkeypatch.setattr("llm.provider.settings.OLLAMA_API_KEY", "ollama-test")
    monkeypatch.setattr("llm.provider.settings.OPENROUTER_API_KEY", "or-test")
    monkeypatch.setattr("llm.provider.settings.GEMINI_API_KEY", "gemini-test")

    import llm.router as router_module
    router_module._deployment_cache.clear()

    def fake_call(self, messages, *args, **kwargs):
        return "pong"

    monkeypatch.setattr("crewai.LLM.call", fake_call)

    results = router_module.check_provider_health()
    assert len(results) > 0
    assert all(r["status"] == "ok" for r in results)
    assert {"agent_name", "provider", "model", "status", "detail"} <= results[0].keys()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `venv/Scripts/python -m pytest tests/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm.router'`

- [ ] **Step 4: Write `llm/router.py`**

```python
"""
Builds, per agent-name alias, the ordered list of real crewai.LLM deployments that
RouterLLM (llm/router_llm.py) falls back across. Reads config/agents_config.yaml's
model_groups section. Also exposes the preflight health check.
"""
from __future__ import annotations

import threading
from typing import Dict, List

import yaml
from crewai import LLM

from config.settings import settings
from llm.provider import LLMConfigError, _PROVIDERS

_config_lock = threading.Lock()
_config_cache: dict | None = None

_deployment_cache: Dict[str, List[LLM]] = {}
_deployment_cache_lock = threading.Lock()


def _load_config() -> dict:
    global _config_cache
    with _config_lock:
        if _config_cache is None:
            with open("config/agents_config.yaml") as f:
                _config_cache = yaml.safe_load(f) or {}
        return _config_cache


def get_deployments(agent_name: str) -> List[LLM]:
    """
    Ordered list of built crewai.LLM instances for one agent-name alias, per
    config/agents_config.yaml's model_groups section. Cached per agent_name.
    Raises LLMConfigError if the alias has no model_groups entry, or if any
    configured deployment is missing its required API key.
    """
    with _deployment_cache_lock:
        if agent_name in _deployment_cache:
            return _deployment_cache[agent_name]

        config = _load_config()
        model_groups = config.get("model_groups", {})
        group = model_groups.get(agent_name)
        if not group:
            raise LLMConfigError(
                f"no model_groups entry for agent_name={agent_name!r} in "
                "config/agents_config.yaml."
            )

        temperature = config.get(agent_name, {}).get("llm_temperature", settings.OLLAMA_TEMPERATURE)

        deployments = []
        for entry in group:
            provider_key = entry["provider"]
            model = entry["model"]
            provider = _PROVIDERS.get(provider_key)
            if provider is None:
                raise LLMConfigError(
                    f"model_groups entry for {agent_name!r} references unknown "
                    f"provider {provider_key!r}. Supported: {', '.join(_PROVIDERS)}"
                )
            deployments.append(provider.build(model, temperature))

        _deployment_cache[agent_name] = deployments
        return deployments


def check_provider_health() -> List[dict]:
    """
    Pings every deployment of every configured agent-name alias with a single
    short call. Computed fresh on each call — no caching, no startup blocking.
    """
    config = _load_config()
    model_groups = config.get("model_groups", {})
    results = []

    for agent_name, group in model_groups.items():
        try:
            deployments = get_deployments(agent_name)
        except LLMConfigError as e:
            for entry in group:
                results.append({
                    "agent_name": agent_name,
                    "provider": entry["provider"],
                    "model": entry["model"],
                    "status": "error",
                    "detail": str(e),
                })
            continue

        for entry, deployment in zip(group, deployments):
            original_timeout = deployment.timeout
            deployment.timeout = 15
            try:
                deployment.call("Reply with the single word: pong")
                results.append({
                    "agent_name": agent_name,
                    "provider": entry["provider"],
                    "model": entry["model"],
                    "status": "ok",
                    "detail": "",
                })
            except Exception as e:
                results.append({
                    "agent_name": agent_name,
                    "provider": entry["provider"],
                    "model": entry["model"],
                    "status": "error",
                    "detail": str(e),
                })
            finally:
                deployment.timeout = original_timeout

    return results
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_router.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add config/agents_config.yaml llm/router.py tests/test_router.py
git commit -m "feat: build LLM deployments from config, add provider health check"
```

---

### Task 4: Rewire `get_llm()` to return a `RouterLLM`

**Files:**
- Modify: `llm/provider.py`
- Test: `tests/test_llm_provider.py`

**Interfaces:**
- Consumes: `llm.router.get_deployments` (Task 3, imported lazily inside `get_llm()` to
  avoid a module-level circular import — `router.py` imports `provider.py` at its top).
  `llm.router_llm.RouterLLM` (Task 2).
- Produces: `llm.provider.get_llm(agent_name: Optional[str]) -> RouterLLM` — same
  signature and import path as before.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_provider.py`:

```python
import pytest

from llm.provider import LLMConfigError, get_llm
from llm.router_llm import RouterLLM


def test_get_llm_returns_router_llm_for_known_alias(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.GROQ_API_KEY", "groq-test")
    monkeypatch.setattr("llm.provider.settings.OLLAMA_API_KEY", "ollama-test")
    monkeypatch.setattr("llm.provider.settings.OPENROUTER_API_KEY", "or-test")
    monkeypatch.setattr("llm.provider.settings.GEMINI_API_KEY", "gemini-test")
    import llm.router as router_module
    router_module._deployment_cache.clear()

    llm = get_llm("researcher")
    assert isinstance(llm, RouterLLM)
    assert len(llm.deployments) == 4


def test_get_llm_unknown_alias_raises():
    with pytest.raises(LLMConfigError):
        get_llm("not_a_real_agent")


def test_get_llm_none_raises():
    with pytest.raises(LLMConfigError):
        get_llm(None)


def test_research_crew_classifies_every_exception_router_llm_can_raise():
    """
    Locks in the design assumption that crew/research_crew.py needs no changes:
    crewai's native per-provider client raises plain openai.* exceptions
    directly (no litellm involved), and _is_provider_unavailable_error already
    has isinstance checks covering every exception type RouterLLM.call() can
    raise. If this ever goes False, RouterLLM's failure modes would silently
    stop being classified as provider-unavailable in the pipeline.
    """
    import httpx
    from crew.research_crew import _is_provider_unavailable_error
    from llm.router_llm import GuardrailRefusalError, _IMMEDIATE_ADVANCE_EXCEPTIONS, _TRANSIENT_EXCEPTIONS

    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(500, request=request)

    for exc_type in (*_TRANSIENT_EXCEPTIONS, *_IMMEDIATE_ADVANCE_EXCEPTIONS):
        if exc_type in (openai.APITimeoutError, openai.APIConnectionError):
            instance = exc_type(request=request)
        else:
            instance = exc_type("test", response=response, body=None)
        assert _is_provider_unavailable_error(instance) is True, exc_type

    assert _is_provider_unavailable_error(GuardrailRefusalError("refused")) is False
```

Add `import openai` to the top of `tests/test_llm_provider.py`, alongside the existing
`import pytest`.

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_llm_provider.py -v`
Expected: FAIL — `get_llm("researcher")` returns a plain `crewai.LLM` today, not a
`RouterLLM`, so the `isinstance` assertion fails.

- [ ] **Step 3: Rewrite `get_llm()` in `llm/provider.py`**

Replace the existing `get_llm` function body (the one reading `_load_agent_overrides()`
and calling `provider.build(...)` directly) with:

```python
def get_llm(agent_name: Optional[str] = None) -> LLM:
    """
    Build the LLM for a given agent name (a top-level key in
    config/agents_config.yaml's model_groups section, e.g. "writer",
    "classifier"). Returns a RouterLLM wrapping that alias's ordered
    deployment list, so calls get conditional fallback across providers.
    """
    if not agent_name:
        raise LLMConfigError("get_llm() requires a non-empty agent_name.")

    from llm.router import get_deployments
    from llm.router_llm import RouterLLM

    deployments = get_deployments(agent_name)
    return RouterLLM(model=agent_name, deployments=deployments)
```

`_load_agent_overrides` and its module-level cache (`_agent_overrides_lock`,
`_agent_overrides_cache`) are now unused — delete them along with this function
rewrite, since `config/agents_config.yaml` reading now lives in `llm/router.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python -m pytest tests/test_llm_provider.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full test suite**

Run: `venv/Scripts/python -m pytest tests/ -v`
Expected: PASS (all tests from Tasks 1-4)

- [ ] **Step 6: Commit**

```bash
git add llm/provider.py tests/test_llm_provider.py
git commit -m "feat: get_llm() returns a RouterLLM with conditional fallback"
```

---

### Task 5: `GET /health/providers` endpoint

**Files:**
- Modify: `app.py`
- Test: `tests/test_health_providers_endpoint.py`

**Interfaces:**
- Consumes: `llm.router.check_provider_health()` (Task 3).
- Produces: `GET /health/providers` → `200 {"providers": [...]}`, same shape
  `check_provider_health()` returns.

- [ ] **Step 1: Write the failing test**

Create `tests/test_health_providers_endpoint.py`:

```python
from fastapi.testclient import TestClient

import app as app_module


def test_health_providers_returns_check_results(monkeypatch):
    fake_results = [
        {"agent_name": "researcher", "provider": "groq", "model": "llama-3.3-70b-versatile",
         "status": "ok", "detail": ""},
        {"agent_name": "researcher", "provider": "gemini", "model": "gemini-2.0-flash",
         "status": "error", "detail": "invalid api key"},
    ]
    monkeypatch.setattr("app.check_provider_health", lambda: fake_results)

    with TestClient(app_module.app) as client:
        response = client.get("/health/providers")

    assert response.status_code == 200
    assert response.json() == {"providers": fake_results}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python -m pytest tests/test_health_providers_endpoint.py -v`
Expected: FAIL with `404 Not Found` (route doesn't exist yet).

- [ ] **Step 3: Add the endpoint to `app.py`**

Add the import near the other top-level imports (after `from eval.scorer import score_report`):

```python
from llm.router import check_provider_health
```

Add the route, next to the existing `@app.get("/health")` route:

```python
@app.get("/health/providers")
def health_providers():
    """Preflight status of every configured LLM deployment, for the UI's status dots."""
    return {"providers": check_provider_health()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python -m pytest tests/test_health_providers_endpoint.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `venv/Scripts/python -m pytest tests/ -v`
Expected: PASS (all tests from Tasks 1-5)

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_health_providers_endpoint.py
git commit -m "feat: add GET /health/providers preflight endpoint"
```

---

### Task 6: Real verification and model-name correction

**Files:** none created — this task edits `config/agents_config.yaml`'s `model_groups`
model names in place, based on real output, and touches no other files.

- [ ] **Step 1: Start the app and call the real health check**

Run: `venv/Scripts/python -m uvicorn app:app --port 8000` (in one terminal), then in
another: `curl http://localhost:8000/health/providers`

- [ ] **Step 2: Fix any deployment reporting `status: "error"`**

For each `error` entry, read its `detail`. A wrong/unavailable model name (e.g.
`gpt-oss:120b-cloud` not present in the account's Ollama Cloud catalog, or an OpenRouter
free-tier slug that has since changed) gets its `model` value in
`config/agents_config.yaml`'s `model_groups` corrected to a real, available model from
that same provider — not swapped to a different provider, since the ordering in the spec
(researcher leads with groq, writer/critic lead with ollama_cloud) is deliberate. A
genuine auth failure (wrong/expired key) is reported back rather than guessed at.

- [ ] **Step 3: Re-run the health check until all deployments report `ok`**

Repeat Steps 1-2 until `curl http://localhost:8000/health/providers` shows
`"status": "ok"` for every entry, or every remaining `error` is independently confirmed
as a real, expected unavailability (e.g., a provider the user hasn't funded).

- [ ] **Step 4: Run one live end-to-end research call**

Run: `venv/Scripts/python -m pytest -k nonexistent` is not applicable here — this is a
real network call, not a pytest test. Instead:

```bash
venv/Scripts/python -c "
from crew.research_crew import ResearchCrew
result = ResearchCrew().run_with_meta('The current state of small modular nuclear reactors', force_rerun=True)
print(result['report'][:500])
print('question_id=', result['question_id'], 'findings=', result['findings_count'])
"
```

Expected: prints a non-empty report excerpt and a `question_id`, with no unhandled
exception — confirms the full pipeline (including the Researcher's tool-calling loop)
still works end to end through `RouterLLM`.

- [ ] **Step 5: Simulate a failover and confirm the run still completes**

Temporarily edit `config/agents_config.yaml`, changing the `researcher` group's first
entry's `model` to an obviously-invalid name (e.g. `"this-model-does-not-exist"`), leaving
the rest of the chain untouched. Re-run the Step 4 command. Expected: the run still
completes and produces a report (the invalid first deployment fails once, `RouterLLM`
advances to the next configured deployment), confirming fallback actually engages under a
real failure rather than only in the mocked unit tests from Task 2. Then revert the edit.

- [ ] **Step 6: Commit the corrected model names (if any changed in Step 2)**

```bash
git add config/agents_config.yaml
git commit -m "fix: correct model_groups entries against real provider availability"
```

If no corrections were needed, skip this commit.
