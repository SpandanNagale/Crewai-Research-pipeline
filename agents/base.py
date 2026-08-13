from typing import Optional

from crewai import LLM

from llm.provider import get_llm as _get_llm


def get_llm(agent_name: Optional[str] = None) -> LLM:
    """Build the LLM for an agent via the LLMProvider factory (llm/provider.py)."""
    return _get_llm(agent_name)
