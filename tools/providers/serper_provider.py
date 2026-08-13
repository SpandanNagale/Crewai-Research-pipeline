from typing import List

import requests

from config.settings import settings
from tools.providers.base import SearchProvider, SearchResult


class SerperSearchProvider(SearchProvider):
    """Google search results via the Serper.dev API. Requires SERPER_API_KEY."""

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        if not settings.SERPER_API_KEY:
            raise RuntimeError("SerperSearchProvider requires SERPER_API_KEY to be set in .env.")

        headers = {
            "X-API-KEY": settings.SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}
        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results: List[SearchResult] = []
        for item in data.get("organic", [])[:num_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                }
            )
        return results
