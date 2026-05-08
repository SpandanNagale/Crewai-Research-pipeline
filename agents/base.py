from crewai import LLM
from config.settings import settings


def get_llm() -> LLM:
    return LLM(
        model=f"ollama/{settings.OLLAMA_MODEL}",
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.OLLAMA_TEMPERATURE,
    )
