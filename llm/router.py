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

_alias_cache: Dict[str, List[str]] = {}


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


def get_deployment_aliases(agent_name: str) -> List[str]:
    """
    Ordered list of config-alias strings (e.g. "groq", "ollama_cloud") parallel to
    get_deployments(agent_name)'s LLM list -- needed because crewai.LLM's own
    `.provider` attribute collapses distinct aliases onto the same value (Groq and
    raw OpenAI both build provider="openai"; ollama_local and ollama_cloud both
    build provider="ollama"), so it can't identify which deployment actually
    served a call. Shares _deployment_cache_lock rather than adding a second lock.
    """
    with _deployment_cache_lock:
        if agent_name in _alias_cache:
            return _alias_cache[agent_name]
        config = _load_config()
        group = config.get("model_groups", {}).get(agent_name)
        if not group:
            raise LLMConfigError(
                f"no model_groups entry for agent_name={agent_name!r} in "
                "config/agents_config.yaml."
            )
        aliases = [entry["provider"] for entry in group]
        _alias_cache[agent_name] = aliases
        return aliases


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
