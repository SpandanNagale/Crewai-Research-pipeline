from typing import List

from tools.providers.base import SearchProvider, SearchResult


class DuckDuckGoSearchProvider(SearchProvider):
    """Free, no-key web search via the `ddgs` package. Default fallback."""

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        try:
            # `duckduckgo_search` was renamed to `ddgs`; the old package is
            # still importable but returns empty results against current
            # DuckDuckGo endpoints, so prefer `ddgs` and fall back only for
            # environments that haven't upgraded.
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError as e:
                raise RuntimeError(
                    "DuckDuckGoSearchProvider requires the 'ddgs' package. "
                    "Install it with: pip install ddgs"
                ) from e

        results: List[SearchResult] = []
        for item in DDGS().text(query, max_results=num_results):
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("body", ""),
                    "url": item.get("href", ""),
                }
            )
        return results
