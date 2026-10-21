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

from llm import response_cache
from observability.context import emit_event
from observability.errors import ErrorCode
from observability.pricing import estimate_cost

logger = logging.getLogger(__name__)

_COOLDOWN_SECONDS = 600.0
_TRANSIENT_RETRY_BACKOFF_SECONDS = 1.0

# Deterministic, low-temperature stages only -- see docs/superpowers's Phase 5
# scope note: "Do not cache the Writer." A stale cached classification or
# comparison is a minor precision loss; a stale cached report is a lie.
_CACHEABLE_ALIASES = frozenset({"classifier", "evidence_comparator"})

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

# Status codes treated as advance-worthy when they surface as the generic
# openai.APIStatusError base class rather than one of the specific subclasses above.
# openai's SDK only maps HTTP 400/401/403/404/409/422/429/5xx to named subclasses (see
# openai._client.OpenAI._make_status_error) — anything else (e.g. Groq's 413 "tokens per
# minute limit exceeded") falls through to the bare base class. Deliberately excludes
# deterministic client-request errors that already have their own specific subclass
# (400 BadRequestError, 409 ConflictError, 422 UnprocessableEntityError) — retrying a
# different deployment won't fix a malformed request, so those propagate uncaught and
# fail the stage instead of cycling through every deployment for an identical error.
_GENERIC_STATUS_ERROR_ADVANCE_CODES = frozenset({408, 413, 503, 529})

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


def unwrap_provider_exception(e: BaseException) -> BaseException:
    """
    The openai.* exception underneath e, if any, else e itself.

    crewai's native provider clients re-raise some openai errors as builtins
    (see crewai.llms.providers.openai.completion): openai.NotFoundError becomes
    ValueError and openai.APIConnectionError becomes ConnectionError, each
    chained via `raise ... from e`. Matching on the raw exception type alone
    misses those, so e.g. an OpenRouter deployment whose free model was pulled
    (a 404) escaped RouterLLM.call() as a bare ValueError without ever trying
    the next deployment.
    """
    seen = set()
    current: Optional[BaseException] = e
    while current is not None and id(current) not in seen:
        if isinstance(current, openai.APIError):
            return current
        seen.add(id(current))
        current = current.__cause__
    return e


def classify_provider_exception(e: Exception) -> ErrorCode:
    e = unwrap_provider_exception(e)
    if isinstance(e, openai.AuthenticationError):
        return ErrorCode.PROVIDER_AUTH_FAILED
    if isinstance(e, openai.RateLimitError):
        return ErrorCode.PROVIDER_RATE_LIMIT
    if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
        return ErrorCode.PROVIDER_TIMEOUT
    if isinstance(e, (openai.PermissionDeniedError, openai.NotFoundError, openai.InternalServerError)):
        return ErrorCode.PROVIDER_UNAVAILABLE
    if isinstance(e, openai.APIStatusError) and e.status_code in _GENERIC_STATUS_ERROR_ADVANCE_CODES:
        return ErrorCode.PROVIDER_UNAVAILABLE
    return ErrorCode.INTERNAL


def _safe_usage(deployment) -> dict:
    try:
        summary = deployment.get_token_usage_summary()
    except Exception:
        return {}
    if hasattr(summary, "prompt_tokens"):
        return {"prompt_tokens": summary.prompt_tokens, "completion_tokens": summary.completion_tokens}
    if isinstance(summary, dict):
        return {"prompt_tokens": summary.get("prompt_tokens"), "completion_tokens": summary.get("completion_tokens")}
    return {}


def _usage_delta(before: dict, after: dict):
    def _diff(key):
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            return None
        return max(a - b, 0)
    return _diff("prompt_tokens"), _diff("completion_tokens")


class RouterLLM(BaseLLM):
    """
    A BaseLLM that fans out to a list of real deployments in order, with
    conditional fallback. `model` is the agent-name alias, for identification/
    logging only — it is never sent to any provider.
    """

    deployments: List[Any]
    aliases: List[str] = []

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
        cache_key = None
        if self.model in _CACHEABLE_ALIASES:
            cache_key = response_cache.make_key(self.model, messages)
            cached = response_cache.get(cache_key)
            if cached is not None:
                tokens_saved = None
                if cached.prompt_tokens is not None or cached.completion_tokens is not None:
                    tokens_saved = (cached.prompt_tokens or 0) + (cached.completion_tokens or 0)
                emit_event(
                    "llm_attempt", model_alias=self.model, resolved_deployment="cache",
                    attempt=0, latency_ms=0.0, cache_hit=True,
                    tokens_saved=tokens_saved, cost_avoided_usd=cached.cost_usd,
                    seconds_saved=cached.latency_ms / 1000, message="cache_hit",
                )
                return cached.result

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
            alias = self.aliases[i] if i < len(self.aliases) else str(i)
            resolved = f"{alias}:{getattr(deployment, 'model', i)}"
            for attempt in range(2):
                started = time.monotonic()
                usage_before = _safe_usage(deployment)
                try:
                    try:
                        result = deployment.call(
                            messages, tools, callbacks, available_functions,
                            from_task, from_agent, response_model,
                        )
                    except Exception as wrapped:
                        # Re-raise the underlying openai error (if crewai wrapped
                        # one) so the except clauses below can classify it.
                        unwrapped = unwrap_provider_exception(wrapped)
                        if unwrapped is wrapped:
                            raise
                        raise unwrapped
                except _TRANSIENT_EXCEPTIONS as e:
                    last_error = e
                    emit_event(
                        "llm_attempt", model_alias=self.model, resolved_deployment=resolved,
                        attempt=attempt, latency_ms=(time.monotonic() - started) * 1000,
                        message=classify_provider_exception(e).value, detail=str(e),
                    )
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
                    emit_event(
                        "llm_attempt", model_alias=self.model, resolved_deployment=resolved,
                        attempt=attempt, latency_ms=(time.monotonic() - started) * 1000,
                        message=classify_provider_exception(e).value, detail=str(e),
                    )
                    logger.warning(
                        "Deployment %s unavailable, advancing immediately: %s",
                        getattr(deployment, "model", i), e,
                    )
                    self._cooldown_until[i] = time.monotonic() + _COOLDOWN_SECONDS
                    break
                except openai.APIStatusError as e:
                    if e.status_code not in _GENERIC_STATUS_ERROR_ADVANCE_CODES:
                        raise
                    last_error = e
                    emit_event(
                        "llm_attempt", model_alias=self.model, resolved_deployment=resolved,
                        attempt=attempt, latency_ms=(time.monotonic() - started) * 1000,
                        message=classify_provider_exception(e).value, detail=str(e),
                    )
                    logger.warning(
                        "Deployment %s unavailable (status %s), advancing immediately: %s",
                        getattr(deployment, "model", i), e.status_code, e,
                    )
                    self._cooldown_until[i] = time.monotonic() + _COOLDOWN_SECONDS
                    break
                else:
                    if isinstance(result, str):
                        _raise_if_refusal(result)
                    usage_after = _safe_usage(deployment)
                    prompt_tok, completion_tok = _usage_delta(usage_before, usage_after)
                    latency_ms = (time.monotonic() - started) * 1000
                    cost_usd = estimate_cost(alias, getattr(deployment, "model", ""), prompt_tok, completion_tok)
                    emit_event(
                        "llm_attempt", model_alias=self.model, resolved_deployment=resolved,
                        attempt=attempt, latency_ms=latency_ms,
                        prompt_tokens=prompt_tok, completion_tokens=completion_tok,
                        cost_usd=cost_usd,
                        message="ok",
                    )
                    if cache_key is not None:
                        response_cache.set(cache_key, response_cache.CachedResponse(
                            result=result, prompt_tokens=prompt_tok, completion_tokens=completion_tok,
                            cost_usd=cost_usd, latency_ms=latency_ms,
                        ))
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

    def supports_function_calling(self) -> bool:
        """
        Not declared on BaseLLM — crewai's executor gates its entire native
        tool-calling path on hasattr(self.llm, "supports_function_calling")
        (see crewai.agents.crew_agent_executor). Without this, every deployment's
        own supports_function_calling() is unreachable and every tool-using agent
        silently falls back to the text-parsed ReAct loop instead of native
        function calling.
        """
        return self.deployments[0].supports_function_calling()
