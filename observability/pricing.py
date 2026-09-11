"""
Static per-token pricing lookup for cost_usd on RunEvent("llm_attempt", ...).
Returns None rather than a fabricated number for any (provider, model) pair
whose real price hasn't been verified against the provider's own pricing
page yet.
"""
from typing import Dict, Optional, Tuple

# (config alias, model) -> (prompt $ / 1K tokens, completion $ / 1K tokens).
# groq/gemini models currently in config/agents_config.yaml are not priced
# here yet -- verify against each provider's live pricing page before adding
# an entry. Until then estimate_cost() correctly returns None for them
# rather than a guess.
_PRICING: Dict[Tuple[str, str], Tuple[float, float]] = {}

_FREE_ALIASES = frozenset({"ollama_local", "ollama_cloud"})


def estimate_cost(
    provider_alias: str,
    model: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Optional[float]:
    if prompt_tokens is None or completion_tokens is None:
        return None
    if provider_alias in _FREE_ALIASES or model.endswith(":free"):
        return 0.0
    rates = _PRICING.get((provider_alias, model))
    if rates is None:
        return None
    prompt_rate, completion_rate = rates
    return (prompt_tokens / 1000) * prompt_rate + (completion_tokens / 1000) * completion_rate
