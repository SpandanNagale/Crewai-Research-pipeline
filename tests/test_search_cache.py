import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import db.session as db_session
from db import repository
from db.models import Base
from observability.bus import EventBus
from observability.context import clear_run, set_stage, start_run
from tools import search_tool as search_tool_module


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    test_session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_session, "SessionLocal", test_session_local)
    EventBus.reset_for_tests()
    yield


def _drain(run_id):
    q = EventBus.instance().queue_sink.get_queue(run_id)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def test_cache_key_normalizes_case_and_whitespace():
    assert search_tool_module._cache_key("  Quantum   Computing ") == "quantum computing"
    assert search_tool_module._cache_key("quantum computing") == "quantum computing"


def test_repository_search_cache_round_trip_and_ttl():
    results = [{"title": "T", "snippet": "S", "url": "http://x"}]
    repository.set_cached_search("k", results, latency_ms=850.0)

    hit = repository.get_cached_search("k", ttl_seconds=3600)
    assert hit == {"results": results, "latency_ms": 850.0}

    expired = repository.get_cached_search("k", ttl_seconds=-1)
    assert expired is None

    assert repository.get_cached_search("missing-key", ttl_seconds=3600) is None


class _CountingProvider:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def search(self, query, num_results=5):
        self.calls += 1
        return self.results


def test_search_tool_cache_hit_skips_provider_and_emits_real_latency(monkeypatch):
    provider = _CountingProvider([{"title": "T", "snippet": "S", "url": "http://x"}])
    monkeypatch.setattr(search_tool_module, "get_search_provider", lambda: provider)

    start_run("run-search-cache")
    set_stage("researcher")

    first = search_tool_module.search_tool.func("Quantum Computing")
    assert provider.calls == 1
    assert "T: S" in first

    second = search_tool_module.search_tool.func("  quantum   computing  ")
    assert provider.calls == 1  # cache hit, provider not called again
    assert first == second

    clear_run()
    cache_events = [e for e in _drain("run-search-cache") if e.event_type == "cache_hit"]
    assert len(cache_events) == 1
    assert cache_events[0].cache_hit is True
    assert cache_events[0].seconds_saved is not None
    assert cache_events[0].seconds_saved >= 0
