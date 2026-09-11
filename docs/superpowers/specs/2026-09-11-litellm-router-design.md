# Phase 1 — LiteLLM Router gateway

Sub-project of `research-pipeline-production-upgrade.md` (Phase 1). Scope: replace
`llm/provider.py`'s hand-rolled per-provider classes with a `litellm.Router`-backed
gateway, without changing the `get_llm(agent_name)` interface any agent module calls.

## Context / resolved conflict

`llm/provider.py`'s docstring says this codebase deliberately avoids litellm, because
`crewai.LLM(model="prefix/model")` validates the string against a vendor-name list and
rejects unrecognized ones — the workaround was passing `provider=` explicitly to skip
that validation. Checked against what's actually installed (crewai 1.14.2): litellm is
not a crewai dependency at all in this version; crewai's `LLM` is a fully native
implementation. So the avoidance was about crewai's *string-based* model resolution
specifically, not a rejection of litellm as a library. A custom `LLM` subclass that
calls `litellm.Router.completion()` directly, bypassing crewai's model-string parsing
entirely, sidesteps the original problem while gaining real fallback/cooldown machinery.
Confirmed `Agent.llm` accepts `Union[str, BaseLLM, None]`, so a `crewai.LLM` subclass is
a valid value — no crewai-side change needed.

## Architecture

- **`llm/router.py`** (new): builds and caches one process-wide `litellm.Router` from
  `config/agents_config.yaml`. Model groups are keyed by the *existing* agent-name
  strings each agent module already passes to `get_llm()`: `planner`, `researcher`,
  `evidence_comparator`, `classifier`, `writer`, `critic`, `evaluator`. No renaming.
  Each group is an ordered `model_list` of deployments (provider + model + credentials).
  Also exposes `check_provider_health() -> dict[str, DeploymentHealth]` for the
  preflight check.
- **`llm/provider.py`**: `get_llm(agent_name)` unchanged in signature. Internally builds
  (and caches per agent_name) a `RouterLLM` instance — a `crewai.LLM` subclass — bound
  to that agent's model group name. `LLMConfigError` semantics are preserved: missing
  key for a *configured* deployment raises a clear, actionable message naming the env
  var, at Router-build time (not silently at call time).
- **`RouterLLM.call()`**: delegates to `router.completion(model=<agent_name>, messages=...)`.
  Converts the litellm response back into the plain string / tool-call shape crewai's
  `LLM.call()` normally returns, so downstream `_extract_pydantic` raw-text parsing is
  untouched. After a successful response, runs the refusal/guardrail check (regex over a
  small set of refusal-shaped phrases — full guardrail detection is Phase 4's job; this
  is the minimal "don't let litellm retry a refusal" hook) — on a hit, raises immediately
  without going through the Router's retry path.

## Deployment provider mapping (litellm model strings)

| provider    | litellm model string          | api_base                          | key env var         |
|-------------|--------------------------------|------------------------------------|----------------------|
| ollama_local| `ollama_chat/<model>`          | `OLLAMA_BASE_URL`                  | none                 |
| ollama_cloud| `openai/<model>`               | `OLLAMA_CLOUD_BASE_URL + "/v1"`    | `OLLAMA_API_KEY`     |
| groq        | `groq/<model>`                 | native (litellm built-in)          | `GROQ_API_KEY`       |
| openrouter  | `openrouter/<model>`           | native (litellm built-in)          | `OPENROUTER_API_KEY` |
| gemini      | `gemini/<model>`               | native (litellm built-in)          | `GEMINI_API_KEY`     |

`ollama_local`'s existing `num_ctx` fix (`extra_body: {"options": {"num_ctx": ...}}`)
carries over via litellm's `extra_body` param passthrough.

## Retry / fallback policy (litellm Router config, not hand-rolled)

- Timeout / 5xx: 1 retry on the *same* deployment, exponential backoff starting at 1s.
- Rate limit (429) / auth failure / model-not-found: 0 retries, advance to next
  deployment in the group immediately.
- Cooldown: 600s after any failure on a deployment (long enough to span one pipeline
  run; benches it for the rest of the run per the doc's requirement).
- Guardrail block / refusal: never retried or failed-over — raised as a distinct
  exception the pipeline stage surfaces directly (fail the stage, not the deployment).

These are `litellm.Router` `RetryPolicy` / `cooldown_time` / `allowed_fails` config,
declared in `llm/router.py`, overridable later without touching call sites.

## Model group ordering (initial defaults — placeholder model names)

Per §4 of the parent doc: Researcher leads with Groq (latency-sensitive, high volume);
Writer and Critic lead with the strongest available model. Everything else defaults
ollama_cloud-primary. Exact model names per deployment are left as reasonable
placeholders and corrected against `GET /health/providers` output rather than guessed
now (wrong guesses burn API quota for nothing):

- `researcher`: groq → ollama_cloud → openrouter → gemini
- `writer`, `critic`: ollama_cloud (strong model) → openrouter → gemini → groq
- `planner`, `classifier`, `evidence_comparator`, `evaluator`: ollama_cloud → groq →
  openrouter → gemini

## New endpoint

`GET /health/providers` — calls `check_provider_health()`, returns per-deployment
`{provider, model, status: ok|error, detail}`. No pipeline run required.

## Config schema addition (`config/agents_config.yaml`)

A new top-level `model_groups:` section, one key per agent-name alias, each a list of
`{provider, model}` deployment entries in fallback order. Existing per-agent
`llm_provider`/`llm_model`/`llm_temperature` override keys are superseded by this (single
source of truth for routing) but `llm_temperature` stays as a plain per-agent override
since temperature isn't provider-specific.

## `.env.example` additions

`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `GEMINI_API_KEY` — documented in the same
style as existing entries.

## Out of scope (later phases)

- Full guardrail taxonomy / `GUARDRAIL_BLOCKED` error code (Phase 4) — this phase only
  adds the minimal hook so refusals don't get laundered through fallback.
- Streaming token-by-token from the resolved deployment (Phase 3).
- Attempt-history persistence to DB / UI rendering (Phase 2/8) — this phase's health
  check is synchronous request/response only, no event bus wiring yet.

## Testing

- Preflight health check run for real against all 5 configured deployments after
  implementation; wrong model names fixed based on actual error output.
- One live `/research` run (small topic) to confirm the pipeline still produces a report
  end to end through the new `RouterLLM` path.
- Simulate a bad deployment (wrong model name for the primary in one group) and confirm
  fallback to the next deployment actually happens and the run still completes.
