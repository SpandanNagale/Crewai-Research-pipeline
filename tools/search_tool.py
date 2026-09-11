import logging
import re
import time

from crewai.tools import tool

from config.settings import settings
from db import repository
from guardrails.tool_output_guardrail import sanitize_untrusted_text, wrap_untrusted_block
from observability.context import emit_event
from tools.providers.base import SearchProvider
from tools.providers.duckduckgo_provider import DuckDuckGoSearchProvider
from tools.providers.serper_provider import SerperSearchProvider

logger = logging.getLogger(__name__)


def _cache_key(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def get_search_provider() -> SearchProvider:
    """
    Resolve the configured SearchProvider.
    SEARCH_PROVIDER=auto (default) picks Serper when a key is configured,
    otherwise falls back to the free, no-key DuckDuckGo provider.
    """
    choice = (settings.SEARCH_PROVIDER or "auto").lower()
    if choice == "serper":
        return SerperSearchProvider()
    if choice == "duckduckgo":
        return DuckDuckGoSearchProvider()
    # auto
    if settings.SERPER_API_KEY:
        return SerperSearchProvider()
    return DuckDuckGoSearchProvider()


def _format(results) -> str:
    if not results:
        return "No results found."
    body = "\n\n".join(
        f"- {sanitize_untrusted_text(r['title'], 'search')}: "
        f"{sanitize_untrusted_text(r['snippet'], 'search')}\n  URL: {r['url']}"
        for r in results
    )
    return wrap_untrusted_block(body, "search results")


@tool("Web Search")
def search_tool(query: str) -> str:
    """Search the web for current information on a topic."""
    cache_key = _cache_key(query)
    cached = repository.get_cached_search(cache_key, settings.SEARCH_CACHE_TTL_SECONDS)
    if cached is not None:
        emit_event(
            "cache_hit",
            message=f"search cache hit for {query!r}",
            cache_hit=True,
            seconds_saved=(cached["latency_ms"] / 1000) if cached["latency_ms"] is not None else None,
        )
        return _format(cached["results"])

    provider = get_search_provider()
    started = time.monotonic()
    try:
        results = provider.search(query, num_results=5)
        repository.set_cached_search(cache_key, results, (time.monotonic() - started) * 1000)
        return _format(results)
    except Exception as e:
        if isinstance(provider, DuckDuckGoSearchProvider):
            logger.error("DuckDuckGo search failed for query %r: %s", query, e)
            return f"Search failed: {e}"
        logger.warning(
            "Search provider %s failed (%s); falling back to DuckDuckGo.",
            type(provider).__name__,
            e,
        )
        try:
            started = time.monotonic()
            results = DuckDuckGoSearchProvider().search(query, num_results=5)
            repository.set_cached_search(cache_key, results, (time.monotonic() - started) * 1000)
            return _format(results)
        except Exception as fallback_error:
            logger.error("DuckDuckGo fallback also failed for query %r: %s", query, fallback_error)
            return f"Search failed: {fallback_error}"


def get_search_tool():
    return search_tool
