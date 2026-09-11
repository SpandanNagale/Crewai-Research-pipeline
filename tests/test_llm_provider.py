import openai
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
