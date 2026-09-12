import openai
import pytest

from llm.byok import BYOKOverride, clear_byok, set_byok
from llm.provider import LLMConfigError, get_llm
from llm.router_llm import RouterLLM


@pytest.fixture(autouse=True)
def _clear_byok_between_tests():
    clear_byok()
    yield
    clear_byok()


def test_get_llm_uses_byok_override_instead_of_configured_model_groups(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.GROQ_API_KEY", "")  # deployment's own key unset
    set_byok(BYOKOverride(provider="groq", api_key="user-supplied-key", model="llama-3.3-70b-versatile"))

    llm = get_llm("researcher")

    assert isinstance(llm, RouterLLM)
    assert len(llm.deployments) == 1
    assert llm.aliases == ["groq"]
    assert llm.deployments[0].api_key == "user-supplied-key"
    assert llm.deployments[0].model == "llama-3.3-70b-versatile"


def test_get_llm_byok_override_raises_for_unknown_provider():
    set_byok(BYOKOverride(provider="not-a-real-provider", api_key="k", model="m"))
    with pytest.raises(LLMConfigError, match="Unknown BYOK provider"):
        get_llm("researcher")


def test_get_llm_ignores_byok_override_once_cleared(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.GROQ_API_KEY", "groq-test")
    monkeypatch.setattr("llm.provider.settings.OLLAMA_API_KEY", "ollama-test")
    monkeypatch.setattr("llm.provider.settings.OPENROUTER_API_KEY", "or-test")
    monkeypatch.setattr("llm.provider.settings.GEMINI_API_KEY", "gemini-test")
    import llm.router as router_module
    router_module._deployment_cache.clear()

    set_byok(BYOKOverride(provider="groq", api_key="k", model="m"))
    clear_byok()

    llm = get_llm("researcher")
    assert len(llm.deployments) == 4  # back to the normal model_groups path


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


def test_get_llm_result_exposes_full_duck_typed_contract_crewai_probes(monkeypatch):
    """
    Regression test for a real bug found in final review: RouterLLM originally
    delegated get_context_window_size/supports_stop_words/supports_multimodal/
    get_token_usage_summary but not supports_function_calling — a method not
    declared on BaseLLM at all, which crewai's agent executor hasattr-probes to
    decide whether to use native tool-calling or fall back to its text-parsed
    ReAct loop. A mocked unit test of RouterLLM in isolation can't catch a
    missing method; this builds a real get_llm() result and checks the whole
    attribute surface crewai's executor actually reads.
    """
    monkeypatch.setattr("llm.provider.settings.GROQ_API_KEY", "groq-test")
    monkeypatch.setattr("llm.provider.settings.OLLAMA_API_KEY", "ollama-test")
    monkeypatch.setattr("llm.provider.settings.OPENROUTER_API_KEY", "or-test")
    monkeypatch.setattr("llm.provider.settings.GEMINI_API_KEY", "gemini-test")
    import llm.router as router_module
    router_module._deployment_cache.clear()

    llm = get_llm("researcher")
    for method_name in (
        "supports_function_calling",
        "get_context_window_size",
        "supports_stop_words",
        "supports_multimodal",
        "get_token_usage_summary",
    ):
        assert hasattr(llm, method_name), f"missing {method_name}"
        assert callable(getattr(llm, method_name)), f"{method_name} not callable"


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
