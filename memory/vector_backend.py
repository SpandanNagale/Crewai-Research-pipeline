"""
Backend-agnostic vector store interface. memory/vector_store.py and
memory/reuse.py are written against this, never against a specific backend's
client object, so the storage engine can change (this project: Chroma ->
pgvector) without touching a single caller.
"""
import abc
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert


@dataclass(frozen=True)
class VectorHit:
    doc_id: str
    document: str
    metadata: dict
    similarity: float  # 0..1, cosine; higher is closer


class VectorBackend(abc.ABC):
    @abc.abstractmethod
    def upsert(self, collection: str, doc_id: str, document: str, metadata: dict) -> None:
        ...

    @abc.abstractmethod
    def search(self, collection: str, query: str, n_results: int) -> List[VectorHit]:
        ...

    @abc.abstractmethod
    def count(self, collection: str) -> int:
        ...

    @abc.abstractmethod
    def clear(self, collection: str) -> None:
        ...


@dataclass
class _StoredDoc:
    document: str
    metadata: dict
    embedding: List[float]


class InMemoryBackend(VectorBackend):
    """
    Same semantics as PgVectorBackend, computed in NumPy over a dict. Used for
    tests and for local dev without Postgres.
    """

    def __init__(self, embedder=None):
        # Imported lazily to avoid a hard dependency on google-genai at
        # import time for callers that supply their own embedder (tests).
        if embedder is None:
            from memory.embedder import embed
            embedder = embed
        self._embed = embedder
        self._docs: Dict[str, Dict[str, _StoredDoc]] = {}

    def upsert(self, collection: str, doc_id: str, document: str, metadata: dict) -> None:
        vectors = self._embed([document])
        if vectors is None:
            return
        self._docs.setdefault(collection, {})[doc_id] = _StoredDoc(
            document=document, metadata=dict(metadata), embedding=vectors[0]
        )

    def search(self, collection: str, query: str, n_results: int) -> List[VectorHit]:
        docs = self._docs.get(collection, {})
        if not docs:
            return []
        vectors = self._embed([query])
        if vectors is None:
            return []
        query_vec = np.array(vectors[0])
        scored = []
        for doc_id, stored in docs.items():
            doc_vec = np.array(stored.embedding)
            denom = np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            similarity = float(np.dot(query_vec, doc_vec) / denom) if denom else 0.0
            scored.append(
                VectorHit(
                    doc_id=doc_id,
                    document=stored.document,
                    metadata=dict(stored.metadata),
                    similarity=similarity,
                )
            )
        scored.sort(key=lambda h: h.similarity, reverse=True)
        return scored[:n_results]

    def count(self, collection: str) -> int:
        return len(self._docs.get(collection, {}))

    def clear(self, collection: str) -> None:
        self._docs.pop(collection, None)


class PgVectorBackend(VectorBackend):
    """
    Production backend: one `vector_documents` table (see db/models.py),
    discriminated by `collection`, queried with pgvector's `<=>` cosine
    operator via db.models.VectorDocument.embedding.cosine_distance(...).
    similarity = 1 - distance, preserving the semantics callers already
    depend on (settings.KNOWLEDGE_REUSE_THRESHOLD is a similarity, not a
    distance).
    """

    def __init__(self, embedder=None):
        if embedder is None:
            from memory.embedder import embed
            embedder = embed
        self._embed = embedder

    def upsert(self, collection: str, doc_id: str, document: str, metadata: dict) -> None:
        from db.models import VectorDocument
        from db.session import get_session

        vectors = self._embed([document])
        if vectors is None:
            return
        with get_session() as session:
            stmt = pg_insert(VectorDocument).values(
                collection=collection,
                doc_id=doc_id,
                document=document,
                metadata_=dict(metadata),
                embedding=vectors[0],
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[VectorDocument.collection, VectorDocument.doc_id],
                set_={
                    "document": stmt.excluded["document"],
                    "metadata": stmt.excluded["metadata"],
                    "embedding": stmt.excluded["embedding"],
                },
            )
            session.execute(stmt)

    def search(self, collection: str, query: str, n_results: int) -> List[VectorHit]:
        from db.models import VectorDocument
        from db.session import get_session

        vectors = self._embed([query])
        if vectors is None:
            return []
        query_vec = vectors[0]
        distance_col = VectorDocument.embedding.cosine_distance(query_vec)
        with get_session() as session:
            rows = session.execute(
                select(VectorDocument, distance_col.label("distance"))
                .where(VectorDocument.collection == collection)
                .order_by(distance_col)
                .limit(n_results)
            ).all()
            return [
                VectorHit(
                    doc_id=row.VectorDocument.doc_id,
                    document=row.VectorDocument.document,
                    metadata=dict(row.VectorDocument.metadata_ or {}),
                    similarity=1 - row.distance,
                )
                for row in rows
            ]

    def count(self, collection: str) -> int:
        from sqlalchemy import func

        from db.models import VectorDocument
        from db.session import get_session

        with get_session() as session:
            return session.execute(
                select(func.count()).select_from(VectorDocument).where(
                    VectorDocument.collection == collection
                )
            ).scalar_one()

    def clear(self, collection: str) -> None:
        from db.models import VectorDocument
        from db.session import get_session

        with get_session() as session:
            session.query(VectorDocument).filter(
                VectorDocument.collection == collection
            ).delete()


def get_backend() -> VectorBackend:
    from config.settings import settings

    if settings.VECTOR_BACKEND == "pgvector":
        return PgVectorBackend()
    return InMemoryBackend()
