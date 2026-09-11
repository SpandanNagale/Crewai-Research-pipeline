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
    # researcher leads with openrouter (native provider), then groq (goes through
    # crewai's openai-compatible client) — see config/agents_config.yaml's comment
    # on the researcher entry for why non-gpt-oss leads this specific chain.
    assert deployments[0].provider == "openrouter"
    assert deployments[1].provider == "openai"


def test_get_deployments_unknown_alias_raises():
    import llm.router as router_module
    router_module._deployment_cache.clear()

    with pytest.raises(LLMConfigError, match="no model_groups entry"):
        router_module.get_deployments("not_a_real_agent")


def test_get_deployments_missing_key_raises_actionable_error(monkeypatch):
    # Blank every key explicitly (not just GROQ_API_KEY) so this is deterministic
    # regardless of which provider leads researcher's chain or what's in a real
    # .env on the machine running it — get_deployments() should raise on the
    # *first* deployment it tries to build, which is whichever alias config says,
    # and every deployment here is equally key-less so the first one always fails.
    monkeypatch.setattr("llm.provider.settings.GROQ_API_KEY", "")
    monkeypatch.setattr("llm.provider.settings.OLLAMA_API_KEY", "")
    monkeypatch.setattr("llm.provider.settings.OPENROUTER_API_KEY", "")
    monkeypatch.setattr("llm.provider.settings.GEMINI_API_KEY", "")
    import llm.router as router_module
    router_module._deployment_cache.clear()

    with pytest.raises(LLMConfigError, match="API_KEY"):
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

    # crewai.LLM is a factory (crewai/llm.py __new__) that dispatches to
    # per-provider native completion classes (OpenAICompletion for
    # groq/ollama_local/ollama_cloud/openrouter since they all go through the
    # OpenAI-compatible client per provider.py's docstring, GeminiCompletion
    # for gemini). None of those subclass crewai.llm.LLM itself, so
    # monkeypatching crewai.LLM.call would not intercept the real .call() that
    # get_deployments()-built instances actually use. Patch the concrete
    # classes instead so the ping in check_provider_health() never reaches
    # the network.
    monkeypatch.setattr(
        "crewai.llms.providers.openai.completion.OpenAICompletion.call", fake_call
    )
    monkeypatch.setattr(
        "crewai.llms.providers.gemini.completion.GeminiCompletion.call", fake_call
    )

    results = router_module.check_provider_health()
    assert len(results) > 0
    assert all(r["status"] == "ok" for r in results)
    assert {"agent_name", "provider", "model", "status", "detail"} <= results[0].keys()


def test_get_deployment_aliases_returns_config_aliases_parallel_to_deployments(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.GROQ_API_KEY", "groq-test")
    monkeypatch.setattr("llm.provider.settings.OLLAMA_API_KEY", "ollama-test")
    monkeypatch.setattr("llm.provider.settings.OPENROUTER_API_KEY", "or-test")
    monkeypatch.setattr("llm.provider.settings.GEMINI_API_KEY", "gemini-test")

    import llm.router as router_module
    router_module._deployment_cache.clear()
    router_module._alias_cache.clear()

    deployments = router_module.get_deployments("researcher")
    aliases = router_module.get_deployment_aliases("researcher")
    assert len(aliases) == len(deployments)
    # Confirms aliases distinguish what deployments[i].provider cannot: Groq and
    # OpenRouter both build distinct crewai provider= strings here, but
    # ollama_local/ollama_cloud would collide on provider="ollama" if both were in
    # one chain -- aliases is the only place that distinction survives.
    assert aliases[0] == "openrouter"
    assert aliases[1] == "groq"
