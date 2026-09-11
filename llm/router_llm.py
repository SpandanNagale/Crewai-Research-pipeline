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
