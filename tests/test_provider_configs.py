import pytest

from llm.provider import (
    GeminiProvider,
    LLMConfigError,
    OpenRouterProvider,
)


def test_openrouter_provider_raises_without_key(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.OPENROUTER_API_KEY", "")
    with pytest.raises(LLMConfigError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider().build("meta-llama/llama-3.3-70b-instruct:free", 0.3)


def test_openrouter_provider_builds_with_key(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(
        "llm.provider.settings.OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    llm = OpenRouterProvider().build("meta-llama/llama-3.3-70b-instruct:free", 0.3)
    assert llm.model == "meta-llama/llama-3.3-70b-instruct:free"
    assert llm.provider == "openrouter"
    assert llm.api_key == "sk-or-test"


def test_gemini_provider_raises_without_key(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.GEMINI_API_KEY", "")
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        GeminiProvider().build("gemini-2.0-flash", 0.3)


def test_gemini_provider_builds_with_key(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.GEMINI_API_KEY", "test-gemini-key")
    llm = GeminiProvider().build("gemini-2.0-flash", 0.3)
    assert llm.model == "gemini-2.0-flash"
    assert llm.provider == "gemini"
    assert llm.api_key == "test-gemini-key"


def test_providers_dict_has_all_five():
    from llm.provider import _PROVIDERS

    assert set(_PROVIDERS.keys()) == {
        "ollama_local",
        "ollama_cloud",
        "groq",
        "openrouter",
        "gemini",
    }
