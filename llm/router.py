"""
Builds, per agent-name alias, the ordered list of real crewai.LLM deployments that
RouterLLM (llm/router_llm.py) falls back across. Reads config/agents_config.yaml's
model_groups section. Also exposes the preflight health check.
"""
from __future__ import annotations

import threading
import logging
from typing import Dict, List, Optional, Tuple

import yaml
from crewai import LLM

from config.settings import settings
from llm.provider import LLMConfigError, _PROVIDERS

_config_lock = threading.Lock()
_config_cache: dict | None = None

_deployment_cache: Dict[str, List[LLM]] = {}
_deployment_cache_lock = threading.Lock()

_alias_cache: Dict[str, List[str]] = {}
_model_cache: Dict[str, List[str]] = {}

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    global _config_cache
    with _config_lock:
        if _config_cache is None:
            with open("config/agents_config.yaml") as f:
                _config_cache = yaml.safe_load(f) or {}
        return _config_cache


def _build_group(agent_name: str) -> Tuple[List[LLM], List[str], List[str]]:
    """
    Build (deployments, aliases, models) for one agent-name alias, skipping any
    deployment whose provider is missing its API key. Callers must hold
    _deployment_cache_lock. Raises LLMConfigError if the alias is unknown,
    references an unknown provider, or if *every* deployment was skipped.
    """
    if agent_name in _deployment_cache:
        return _deployment_cache[agent_name], _alias_cache[agent_name], _model_cache[agent_name]

    config = _load_config()
    group = config.get("model_groups", {}).get(agent_name)
    if not group:
        raise LLMConfigError(
            f"no model_groups entry for agent_name={agent_name!r} in "
            "config/agents_config.yaml."
        )

    temperature = config.get(agent_name, {}).get("llm_temperature", settings.OLLAMA_TEMPERATURE)

    deployments: List[LLM] = []
    aliases: List[str] = []
    models: List[str] = []
    skipped: List[str] = []
    for entry in group:
        provider_key = entry["provider"]
        model = entry["model"]
        provider = _PROVIDERS.get(provider_key)
        if provider is None:
            raise LLMConfigError(
                f"model_groups entry for {agent_name!r} references unknown "
                f"provider {provider_key!r}. Supported: {', '.join(_PROVIDERS)}"
            )
        try:
            deployments.append(provider.build(model, temperature))
        except LLMConfigError as e:
            # A missing key for one provider shouldn't take the whole agent
            # down while other providers in the chain are configured.
            skipped.append(str(e))
            logger.warning("Skipping %s:%s for %r: %s", provider_key, model, agent_name, e)
            continue
        aliases.append(provider_key)
        models.append(model)

    if not deployments:
        raise LLMConfigError(
            f"no usable LLM deployment for agent_name={agent_name!r}: every provider "
            "in its model_groups chain is missing its API key. Set at least one of "
            "GROQ_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY. Details: "
            + " | ".join(skipped)
        )

    _deployment_cache[agent_name] = deployments
    _alias_cache[agent_name] = aliases
    _model_cache[agent_name] = models
    return deployments, aliases, models


def get_deployments(agent_name: str) -> List[LLM]:
    """
    Ordered list of built crewai.LLM instances for one agent-name alias, per
    config/agents_config.yaml's model_groups section. Cached per agent_name.
    Deployments whose provider has no API key configured are skipped; raises
    LLMConfigError if the alias has no model_groups entry or no deployment
    is usable at all.
    """
    with _deployment_cache_lock:
        return _build_group(agent_name)[0]


def get_deployment_aliases(agent_name: str) -> List[str]:
    """
    Ordered list of config-alias strings (e.g. "groq", "gemini") parallel to
    get_deployments(agent_name)'s LLM list -- needed because crewai.LLM's own
    `.provider` attribute collapses distinct aliases onto the same value (Groq and
    raw OpenAI both build provider="openai"; ollama_local and ollama_cloud both
    build provider="ollama"), so it can't identify which deployment actually
    served a call. Skipped (key-less) deployments are omitted here too, so the
    two lists always stay index-aligned.
    """
    with _deployment_cache_lock:
        return _build_group(agent_name)[1]


def find_model_for_provider(provider_key: str) -> Optional[str]:
    """
    A reasonable default model for a BYOK request (llm/byok.py): reuses
    whatever model this deployment already trusts that provider with for some
    agent, rather than hardcoding a second, easily-stale model list. None if
    no agent's model_groups lists this provider at all.
    """
    config = _load_config()
    for group in config.get("model_groups", {}).values():
        for entry in group:
            if entry["provider"] == provider_key:
                return entry["model"]
    return None


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
            with _deployment_cache_lock:
                deployments, aliases, models = _build_group(agent_name)
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

        # Report providers skipped for a missing key, so /health/providers
        # still shows the full configured chain.
        built = set(zip(aliases, models))
        for entry in group:
            if (entry["provider"], entry["model"]) not in built:
                results.append({
                    "agent_name": agent_name,
                    "provider": entry["provider"],
                    "model": entry["model"],
                    "status": "skipped",
                    "detail": "API key not configured",
                })

        for alias, model, deployment in zip(aliases, models, deployments):
            entry = {"provider": alias, "model": model}
            # Ping a throwaway copy, never the cached instance: get_deployments()
            # returns the exact same LLM objects live RouterLLMs hold, and a health
            # check firing during a real run must not mutate a deployment's timeout
            # out from under an in-flight generation. Not every native provider
            # completion class carries a `timeout` field (e.g. GeminiCompletion
            # doesn't) — pydantic allows the assignment either way, it's just a
            # no-op on the copy for those.
            probe = deployment.model_copy()
            probe.timeout = 15
            try:
                probe.call("Reply with the single word: pong")
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

    return results
