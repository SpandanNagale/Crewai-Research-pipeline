"""
Thin per-collection wrapper over memory/vector_backend.py::VectorBackend.
Callers (memory/reuse.py) never see the underlying backend or its client --
this is the only file that knows which backend is selected.
"""
from typing import List

from memory.vector_backend import VectorHit, get_backend


class ResearchMemory:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._backend = get_backend()

    def upsert(self, doc_id: str, document: str, metadata: dict) -> None:
        self._backend.upsert(self.collection_name, doc_id, document, metadata)

    def search(self, query: str, n_results: int) -> List[VectorHit]:
        return self._backend.search(self.collection_name, query, n_results)

    def count(self) -> int:
        return self._backend.count(self.collection_name)

    def clear(self) -> None:
        self._backend.clear(self.collection_name)
