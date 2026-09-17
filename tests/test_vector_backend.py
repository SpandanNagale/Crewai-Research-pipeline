"""
Same test body run against both VectorBackend implementations so the
in-memory backend (tests, local dev) can't silently drift from pgvector's
real semantics. The pgvector variant is skipped when TEST_DATABASE_URL
isn't configured (no Postgres available in this environment).
"""
import hashlib
import os
import uuid

import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import db.session as db_session
from db.models import Base
from memory.vector_backend import InMemoryBackend, PgVectorBackend

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


def _fake_embed(texts):
    """
    Deterministic, content-based, real cosine math -- no network call.
    768-dim to match the schema's vector(768) column exactly.
    """
    if texts is None:
        return None
    vectors = []
    for text in texts:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=768)
        vectors.append((v / np.linalg.norm(v)).tolist())
    return vectors


@pytest.fixture(params=["memory", "pgvector"])
def backend(request, monkeypatch):
    if request.param == "pgvector":
        if not TEST_DATABASE_URL:
            pytest.skip("TEST_DATABASE_URL not configured, skipping pgvector backend tests")
        engine = create_engine(TEST_DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        Base.metadata.create_all(engine)
        test_session_local = sessionmaker(bind=engine, expire_on_commit=False)
        monkeypatch.setattr(db_session, "SessionLocal", test_session_local)
        return PgVectorBackend(embedder=_fake_embed)
    return InMemoryBackend(embedder=_fake_embed)


@pytest.fixture
def collection():
    return f"test-{uuid.uuid4().hex}"


def test_upsert_then_search_finds_the_document(backend, collection):
    backend.upsert(collection, "doc-1", "apple", {"k": "v"})

    hits = backend.search(collection, "apple", n_results=1)

    assert len(hits) == 1
    assert hits[0].doc_id == "doc-1"
    assert hits[0].document == "apple"
    assert hits[0].metadata == {"k": "v"}
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-6)


def test_search_ranks_the_closer_match_first(backend, collection):
    backend.upsert(collection, "a", "apple", {})
    backend.upsert(collection, "b", "car", {})

    hits = backend.search(collection, "apple", n_results=2)

    assert hits[0].doc_id == "a"
    assert hits[0].similarity >= hits[1].similarity


def test_count_reflects_number_of_upserts(backend, collection):
    assert backend.count(collection) == 0
    backend.upsert(collection, "a", "apple", {})
    backend.upsert(collection, "b", "car", {})
    assert backend.count(collection) == 2


def test_upsert_same_doc_id_overwrites_not_duplicates(backend, collection):
    backend.upsert(collection, "a", "apple", {"v": 1})
    backend.upsert(collection, "a", "apple pie", {"v": 2})

    assert backend.count(collection) == 1
    hits = backend.search(collection, "apple pie", n_results=1)
    assert hits[0].document == "apple pie"
    assert hits[0].metadata == {"v": 2}


def test_search_on_empty_collection_returns_empty_list_not_error(backend, collection):
    assert backend.search(collection, "anything", n_results=5) == []


def test_search_respects_n_results_limit(backend, collection):
    for i in range(5):
        backend.upsert(collection, f"doc-{i}", f"text-{i}", {})

    hits = backend.search(collection, "text-0", n_results=2)

    assert len(hits) == 2


def test_clear_removes_all_documents_in_collection(backend, collection):
    backend.upsert(collection, "a", "apple", {})

    backend.clear(collection)

    assert backend.count(collection) == 0


def test_clear_does_not_affect_other_collections(backend, collection):
    other = collection + "-other"
    backend.upsert(collection, "a", "apple", {})
    backend.upsert(other, "b", "car", {})

    backend.clear(collection)

    assert backend.count(collection) == 0
    assert backend.count(other) == 1


def test_metadata_supports_a_real_list_not_just_scalars(backend, collection):
    # The whole point of moving off Chroma: no more comma-join/split hack
    # for list-valued metadata (see memory/reuse.py's finding_ids).
    backend.upsert(collection, "a", "apple", {"finding_ids": [1, 2, 3]})

    hits = backend.search(collection, "apple", n_results=1)

    assert hits[0].metadata["finding_ids"] == [1, 2, 3]


def test_upsert_swallows_embedding_failure(collection):
    backend = InMemoryBackend(embedder=lambda texts: None)

    backend.upsert(collection, "a", "apple", {})  # must not raise

    assert backend.count(collection) == 0


def test_search_swallows_embedding_failure(collection):
    backend = InMemoryBackend(embedder=lambda texts: None)

    assert backend.search(collection, "apple", n_results=1) == []


def test_knowledge_reuse_checker_degrades_gracefully_when_embedding_fails(monkeypatch):
    monkeypatch.setattr("config.settings.settings.VECTOR_BACKEND", "memory")
    monkeypatch.setattr("memory.embedder.embed", lambda texts: None)

    from memory.reuse import KnowledgeReuseChecker

    checker = KnowledgeReuseChecker()

    assert checker.find_similar("some topic") is None  # must not raise
    checker.index_question(1, "some topic", "conclusion")  # must not raise
