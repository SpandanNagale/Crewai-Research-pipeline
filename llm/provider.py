"""
LLM provider abstraction.

CrewAI agents get their LLM instance from get_llm(agent_name), which builds a
RouterLLM (llm/router_llm.py) wrapping the ordered list of deployments for
that agent-name alias, as configured in config/agents_config.yaml's
model_groups section (built by llm/router.py). Each deployment is a
crewai.LLM built by one of the provider classes below.

Implementation note: this installation does not depend on litellm. crewai
1.14+ ships native provider clients and routes "ollama"/"openai" through its
own OpenAI-compatible HTTP client when you pass provider=... explicitly, so
Ollama Cloud and Groq are wired up as base_url + api_key swaps against that
native client rather than through litellm's "groq/<model>" routing (which
would require the optional litellm package). Passing provider= explicitly
(rather than a "prefix/model" string) skips crewai's model-name validation,
which only recognizes vendor model names, and always resolves to the native
client we want.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from crewai import LLM

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMConfigError(RuntimeError):
    """Raised when a configured LLM provider is missing required config (e.g. an API key)."""


class LLMProvider(ABC):
    """One pluggable LLM backend."""

    @abstractmethod
    def build(self, model: str, temperature: float, api_key: Optional[str] = None) -> LLM:
        """api_key overrides the provider's configured settings.X_API_KEY -- used
        for a bring-your-own-key request (see llm/byok.py). None means "use the
        deployment's own configured key", the existing behavior."""
        ...


class OllamaLocalProvider(LLMProvider):
    """Local Ollama server. Zero-cost, no API key required. Default provider."""

    def build(self, model: str, temperature: float, api_key: Optional[str] = None) -> LLM:
        return LLM(
            model=model,
            provider="ollama",
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
            # Ollama's OpenAI-compatible endpoint ignores context length unless
            # told explicitly; without this it silently serves at its default
            # (~4k tokens), which structured JSON outputs (research findings,
            # contradictions) can overflow mid-generation. additional_params
            # is merged verbatim into the request; "extra_body" is the
            # openai-python escape hatch for non-standard fields like Ollama's
            # "options".
            additional_params={"extra_body": {"options": {"num_ctx": settings.OLLAMA_NUM_CTX}}},
        )


class OllamaCloudProvider(LLMProvider):
    """Ollama Cloud. Same API surface as local Ollama, just base_url + api_key."""

    def build(self, model: str, temperature: float, api_key: Optional[str] = None) -> LLM:
        key = api_key or settings.OLLAMA_API_KEY
        if not key:
            raise LLMConfigError(
                "LLM_PROVIDER=ollama_cloud requires OLLAMA_API_KEY to be set in .env "
                "(get one at https://ollama.com after `ollama signin`)."
            )
        return LLM(
            model=model,
            provider="ollama",
            base_url=settings.OLLAMA_CLOUD_BASE_URL,
            api_key=key,
            temperature=temperature,
            additional_params={"extra_body": {"options": {"num_ctx": settings.OLLAMA_NUM_CTX}}},
        )


class GroqProvider(LLMProvider):
    """Groq's OpenAI-compatible API. Fast inference, free tier available."""

    def build(self, model: str, temperature: float, api_key: Optional[str] = None) -> LLM:
        key = api_key or settings.GROQ_API_KEY
        if not key:
            raise LLMConfigError(
                "LLM_PROVIDER=groq requires GROQ_API_KEY to be set in .env "
                "(get one at https://console.groq.com/keys)."
            )
        return LLM(
            model=model,
            provider="openai",
            base_url=settings.GROQ_BASE_URL,
            api_key=key,
            temperature=temperature,
        )


class OpenRouterProvider(LLMProvider):
    """OpenRouter — aggregates many hosted models behind one OpenAI-compatible API."""

    def build(self, model: str, temperature: float, api_key: Optional[str] = None) -> LLM:
        key = api_key or settings.OPENROUTER_API_KEY
        if not key:
            raise LLMConfigError(
                "OPENROUTER_API_KEY is required to use the openrouter provider "
                "(get one at https://openrouter.ai/keys)."
            )
        return LLM(
            model=model,
            provider="openrouter",
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=key,
            temperature=temperature,
        )


class GeminiProvider(LLMProvider):
    """Google Gemini. crewai has a native client for this provider — no base_url needed."""

    def build(self, model: str, temperature: float, api_key: Optional[str] = None) -> LLM:
        key = api_key or settings.GEMINI_API_KEY
        if not key:
            raise LLMConfigError(
                "GEMINI_API_KEY is required to use the gemini provider "
                "(get one at https://aistudio.google.com/apikey)."
            )
        return LLM(
            model=model,
            provider="gemini",
            api_key=key,
            temperature=temperature,
        )


_PROVIDERS = {
    "ollama_local": OllamaLocalProvider(),
    "ollama_cloud": OllamaCloudProvider(),
    "groq": GroqProvider(),
    "openrouter": OpenRouterProvider(),
    "gemini": GeminiProvider(),
}


def get_llm(agent_name: Optional[str] = None) -> LLM:
    """
    Build the LLM for a given agent name (a top-level key in
    config/agents_config.yaml's model_groups section, e.g. "writer",
    "classifier"). Returns a RouterLLM wrapping that alias's ordered
    deployment list, so calls get conditional fallback across providers.
    """
    if not agent_name:
        raise LLMConfigError("get_llm() requires a non-empty agent_name.")

    from llm.byok import get_byok
    from llm.router import get_deployment_aliases, get_deployments
    from llm.router_llm import RouterLLM

    override = get_byok()
    if override is not None:
        provider = _PROVIDERS.get(override.provider)
        if provider is None:
            raise LLMConfigError(
                f"Unknown BYOK provider {override.provider!r}. Supported: {', '.join(_PROVIDERS)}"
            )
        deployment = provider.build(override.model, settings.OLLAMA_TEMPERATURE, api_key=override.api_key)
        return RouterLLM(model=agent_name, deployments=[deployment], aliases=[override.provider])

    deployments = get_deployments(agent_name)
    aliases = get_deployment_aliases(agent_name)
    return RouterLLM(model=agent_name, deployments=deployments, aliases=aliases)
