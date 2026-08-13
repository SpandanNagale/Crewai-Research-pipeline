from abc import ABC, abstractmethod
from typing import List, TypedDict


class SearchResult(TypedDict):
    title: str
    snippet: str
    url: str


class SearchProvider(ABC):
    """One pluggable web search backend."""

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        ...
