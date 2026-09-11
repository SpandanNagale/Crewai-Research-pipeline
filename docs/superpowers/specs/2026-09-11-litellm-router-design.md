# Phase 1 — LLM gateway: conditional fallback across 5 providers

Sub-project of `research-pipeline-production-upgrade.md` (Phase 1). Scope: give
`llm/provider.py` conditional fallback (timeout/5xx retry-then-advance, rate-limit/auth
advance-immediately, cooldowns, never-retry-a-refusal) across ollama_local, ollama_cloud,
groq, openrouter, and gemini, without changing the `get_llm(agent_name)` interface any
agent module calls.

## Revision note

This spec originally called for `litellm.Router`, per the parent doc's literal wording.
Rejected after inspecting the installed crewai (1.14.2) source: `crewai.LLM.call()` runs
the Researcher's entire tool-execution loop internally
(`_handle_non_streaming_response(..., available_functions=...)`), plus the Ollama
`num_ctx` fix and the raw-JSON structured-output parsing this codebase deliberately uses
instead of `output_pydantic`. Routing the real request through `litellm.Router.completion()`
would bypass all three — a correctness regression, not a refactor. Confirmed instead that
`crewai.llms.base_llm.BaseLLM` (crewai 1.14.2) is *designed* for this: its docstring reads
"Users can extend this class to create custom LLM implementations that don't rely on
litellm's authentication mechanism," and `call` is its only `@abstractmethod`. So:
hand-roll the fallback/cooldown state machine, no litellm dependency.

## Architecture

- **`llm/provider.py`**: keeps the existing `LLMProvider` ABC and its `.build(model,
  temperature) -> LLM` classes (`OllamaLocalProvider`, `OllamaCloudProvider`,
  `GroqProvider`, unchanged), adding two more of the same shape:
  - `OpenRouterProvider` — `LLM(model=model, provider="openrouter", base_url=settings.OPENROUTER_BASE_URL, api_key=settings.OPENROUTER_API_KEY, temperature=temperature)`.
    `"openrouter"` is in crewai's `SUPPORTED_NATIVE_PROVIDERS` (OpenAI-compatible).
  - `GeminiProvider` — `LLM(model=model, provider="gemini", api_key=settings.GEMINI_API_KEY, temperature=temperature)`.
    crewai 1.14.2 has a native `GeminiCompletion` client for `provider="gemini"` — no
    `base_url` needed.
  - Each still raises `LLMConfigError` (unchanged semantics) when its required key is
    missing, naming the env var and where to get it, matching the existing two providers.
- **`llm/router.py`** (new): reads `config/agents_config.yaml`'s new `model_groups:`
  section, and for each agent-name alias (`planner`, `researcher`,
  `evidence_comparator`, `classifier`, `writer`, `critic`, `evaluator`) builds an ordered
  list of real `crewai.LLM` instances — one per `{provider, model}` entry — via the
  provider classes above. Exposes `get_deployments(agent_name) -> list[LLM]` (cached per
  agent_name) and `check_provider_health() -> list[DeploymentHealth]`.
- **`llm/provider.py::get_llm(agent_name)`**: unchanged signature. Returns a `RouterLLM`
  (see below) built from `router.get_deployments(agent_name)`.
- **`llm/router_llm.py`** (new): `RouterLLM(BaseLLM)` — the fallback/cooldown state
  machine. Holds `deployments: list[LLM]` (pydantic field) and two private, non-pydantic
  attrs: `_cooldown_until: dict[int, float]` (deployment index → `time.monotonic()`
  deadline) and nothing else — no other mutable cross-call state.

### `RouterLLM.call()`

```python
def call(self, messages, tools=None, callbacks=None, available_functions=None,
         from_task=None, from_agent=None, response_model=None):
    now = time.monotonic()
    order = [i for i in range(len(self.deployments)) if self._cooldown_until.get(i, 0.0) <= now]
    if not order:  # everything cooling down — try the original order anyway
        order = list(range(len(self.deployments)))

    last_error: Exception | None = None
    for i in order:
        deployment = self.deployments[i]
        for attempt in range(2):  # 1 retry on timeout/5xx, same deployment
            try:
                result = deployment.call(messages, tools, callbacks, available_functions,
                                          from_task, from_agent, response_model)
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
                last_error = e
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                self._cooldown_until[i] = time.monotonic() + 600
                break
            except (openai.RateLimitError, openai.AuthenticationError,
                     openai.PermissionDeniedError, openai.NotFoundError) as e:
                last_error = e
                self._cooldown_until[i] = time.monotonic() + 600
                break
            else:
                if isinstance(result, str):
                    _raise_if_refusal(result)  # raises GuardrailRefusalError, uncaught here
                return result
    raise last_error or LLMConfigError("no deployments configured")
```

`GuardrailRefusalError` (new, `llm/router_llm.py`) is not in any `except` clause above, so
a refusal propagates out of `call()` immediately — never retried, never failed over.
`_raise_if_refusal` is a small case-insensitive check against a short fixed list of
refusal-opening phrases ("i cannot assist", "i can't help with that", "i'm not able to
provide", "as an ai language model, i cannot", "i won't provide", "i refuse to"). This is
a minimal placeholder hook, not the real guardrail taxonomy — Phase 4 replaces it.

`RouterLLM` delegates the handful of `BaseLLM` methods CrewAI's executor actually reads —
`get_context_window_size()`, `supports_stop_words()`, `supports_multimodal()`,
`get_token_usage_summary()` — to `self.deployments[0]` (the primary), since these describe
model *capabilities* used before a call is made, not the outcome of a specific attempt.

## Retry / fallback policy (hand-rolled, matches parent doc §4 exactly)

- Timeout / 5xx (`APITimeoutError`, `APIConnectionError`, `InternalServerError`): 1 retry
  on the *same* deployment after a 1s backoff, then advance.
- Rate limit / auth failure / model-not-found (`RateLimitError`, `AuthenticationError`,
  `PermissionDeniedError`, `NotFoundError`): 0 retries, advance immediately.
- Cooldown: 600s from time of failure (benches a deployment for the rest of a typical
  pipeline run, per-`RouterLLM`-instance — each agent's own fallback chain, not global).
- Guardrail / refusal: never retried, never failed over — raised straight out of `call()`.

These are plain `except` clauses in `RouterLLM.call()`, not external config — acceptable
per the parent doc's intent (conditional fallback keyed on error class); "config-driven"
in the doc referred to *which models* are in each chain (`config/agents_config.yaml`),
not the retry policy's error-class semantics.

## Model group ordering (initial defaults — placeholder model names)

Per §4 of the parent doc: Researcher leads with Groq (latency-sensitive, high volume);
Writer and Critic lead with the strongest available model. Everything else defaults
ollama_cloud-primary (matches how the user characterized that key: "ollama primary
provider"). Exact model names per deployment are placeholders, corrected against
`GET /health/providers` output after implementation rather than guessed now:

- `researcher`: groq → ollama_cloud → openrouter → gemini
- `writer`, `critic`: ollama_cloud (strong model) → openrouter → gemini → groq
- `planner`, `classifier`, `evidence_comparator`, `evaluator`: ollama_cloud → groq →
  openrouter → gemini

## New endpoint

`GET /health/providers` — for every alias in `model_groups`, pings each deployment with
a single short call (`deployment.call("Reply with the single word: pong")`, `timeout`
temporarily set to 15s on that `LLM` instance and restored after), computed fresh on each
request (no caching, no startup blocking — see Out of scope). Returns
`[{agent_name, provider, model, status: "ok"|"error", detail}]`.

## Config schema addition (`config/agents_config.yaml`)

New top-level `model_groups:` section, one key per agent-name alias, each an ordered list
of `{provider, model}` deployment entries — single source of truth for routing, replacing
the commented-out `llm_provider`/`llm_model` per-agent override example block (removed).
`llm_temperature` stays as a plain per-agent override (not provider-specific).

## `config/settings.py` / `.env.example` additions

`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`),
`GEMINI_API_KEY` — same style as existing entries.

## Out of scope (later phases)

- Full guardrail taxonomy / `GUARDRAIL_BLOCKED` error code (Phase 4) — this phase only
  adds the minimal `_raise_if_refusal` hook so refusals don't get laundered through
  fallback.
- Streaming token-by-token from the resolved deployment (Phase 3).
- Attempt-history persistence to DB / UI rendering (Phase 2/8) — no event bus wiring yet.
- Eager health check at process startup with caching — `GET /health/providers` computes
  fresh per request instead, to avoid uncertain startup-blocking behavior from a slow or
  hung provider. Revisit once Phase 2's event bus exists to run it as a background task.

## Testing

- Unit tests (no network): `RouterLLM.call()`'s fallback/retry/cooldown logic, exercised
  against fake deployments (stub objects with a scripted `.call()`) covering: same-
  deployment retry then advance on timeout, immediate advance on rate-limit/auth,
  cooldown benching a deployment across two calls, refusal raised uncaught, all-deployments-
  exhausted raises the last real error.
- Unit tests: each new provider class raises `LLMConfigError` with the right env var name
  when its key is missing; builds a valid `LLM` when present.
- Confirmed (no code change needed): `crew/research_crew.py::_is_provider_unavailable_error`
  already classifies every exception type `RouterLLM.call()` can raise, since crewai's
  native per-provider client raises plain `openai.*` exceptions directly — no litellm
  involved anywhere in this call path.
- Manual, after implementation: run `GET /health/providers` for real against all 5
  configured deployments; fix any wrong model names based on actual error output. One
  live `/research` run on a small topic end to end. One run with a deliberately wrong
  model name on a chain's primary deployment, confirming fallback to the next deployment
  actually completes the run.
