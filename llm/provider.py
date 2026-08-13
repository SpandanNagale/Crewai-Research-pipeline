"""
LLM provider abstraction.

CrewAI agents get their LLM instance from get_llm(agent_name) instead of
constructing crewai.LLM directly. The provider (ollama_local / ollama_cloud /
groq), model, and temperature are resolved from config/agents_config.yaml
per-agent overrides, falling back to the global LLM_PROVIDER / LLM_MODEL in
.env.

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
import os
import threading
from abc import ABC, abstractmethod
from typing import Optional

import yaml
from crewai import LLM

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMConfigError(RuntimeError):
    """Raised when a configured LLM provider is missing required config (e.g. an API key)."""


class LLMProvider(ABC):
    """One pluggable LLM backend."""

    @abstractmethod
    def build(self, model: str, temperature: float) -> LLM:
        ...


class OllamaLocalProvider(LLMProvider):
    """Local Ollama server. Zero-cost, no API key required. Default provider."""

    def build(self, model: str, temperature: float) -> LLM:
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

    def build(self, model: str, temperature: float) -> LLM:
        if not settings.OLLAMA_API_KEY:
            raise LLMConfigError(
                "LLM_PROVIDER=ollama_cloud requires OLLAMA_API_KEY to be set in .env "
                "(get one at https://ollama.com after `ollama signin`)."
            )
        os.environ.setdefault("OLLAMA_API_KEY", settings.OLLAMA_API_KEY)
        return LLM(
            model=model,
            provider="ollama",
            base_url=settings.OLLAMA_CLOUD_BASE_URL,
            api_key=settings.OLLAMA_API_KEY,
            temperature=temperature,
            additional_params={"extra_body": {"options": {"num_ctx": settings.OLLAMA_NUM_CTX}}},
        )


class GroqProvider(LLMProvider):
    """Groq's OpenAI-compatible API. Fast inference, free tier available."""

    def build(self, model: str, temperature: float) -> LLM:
        if not settings.GROQ_API_KEY:
            raise LLMConfigError(
                "LLM_PROVIDER=groq requires GROQ_API_KEY to be set in .env "
                "(get one at https://console.groq.com/keys)."
            )
        os.environ.setdefault("OPENAI_API_KEY", settings.GROQ_API_KEY)
        return LLM(
            model=model,
            provider="openai",
            base_url=settings.GROQ_BASE_URL,
            api_key=settings.GROQ_API_KEY,
            temperature=temperature,
        )


_PROVIDERS = {
    "ollama_local": OllamaLocalProvider(),
    "ollama_cloud": OllamaCloudProvider(),
    "groq": GroqProvider(),
}

_agent_overrides_lock = threading.Lock()
_agent_overrides_cache: Optional[dict] = None


def _load_agent_overrides() -> dict:
    global _agent_overrides_cache
    with _agent_overrides_lock:
        if _agent_overrides_cache is None:
            try:
                with open("config/agents_config.yaml") as f:
                    _agent_overrides_cache = yaml.safe_load(f) or {}
            except FileNotFoundError:
                _agent_overrides_cache = {}
        return _agent_overrides_cache


def get_llm(agent_name: Optional[str] = None) -> LLM:
    """
    Build the LLM for a given agent name (a top-level key in
    config/agents_config.yaml, e.g. "writer", "classifier"). Falls back to
    the global LLM_PROVIDER / LLM_MODEL / OLLAMA_TEMPERATURE from .env when
    the agent has no override, or when agent_name is None.
    """
    provider_name = settings.LLM_PROVIDER
    model = settings.LLM_MODEL
    temperature = settings.OLLAMA_TEMPERATURE

    if agent_name:
        cfg = _load_agent_overrides().get(agent_name, {}) or {}
        provider_name = cfg.get("llm_provider", provider_name)
        model = cfg.get("llm_model", model)
        temperature = cfg.get("llm_temperature", temperature)

    provider = _PROVIDERS.get(provider_name)
    if provider is None:
        supported = ", ".join(_PROVIDERS.keys())
        raise LLMConfigError(
            f"Unknown LLM_PROVIDER '{provider_name}' (agent={agent_name!r}). "
            f"Supported: {supported}"
        )

    try:
        return provider.build(model, temperature)
    except LLMConfigError as e:
        logger.error("LLM configuration error for agent '%s' (provider=%s): %s", agent_name, provider_name, e)
        raise
