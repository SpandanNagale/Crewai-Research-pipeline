import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import db.session as db_session
from crew.research_crew import ResearchCrew
from db import repository
from db.models import Base
from observability.bus import EventBus
from observability.context import clear_run, set_stage, start_run


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    test_session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_session, "SessionLocal", test_session_local)
    EventBus.reset_for_tests()
    yield


class _FakeReuseChecker:
    def __init__(self, hit):
        self._hit = hit
        self.indexed = []

    def find_similar_subquestion(self, sub_question):
        return self._hit

    def index_sub_question(self, sub_question, finding_ids, latency_ms, prompt_tokens, completion_tokens, cost_usd):
        self.indexed.append((sub_question, finding_ids, latency_ms, prompt_tokens, completion_tokens, cost_usd))


def _drain(run_id):
    q = EventBus.instance().queue_sink.get_queue(run_id)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def test_subquestion_cache_hit_copies_findings_without_calling_researcher(monkeypatch):
    source_question_id = repository.create_question("prior topic")
    source_id = repository.get_or_create_source(source_question_id, "http://x", "Title X")
    finding_id = repository.add_finding(source_question_id, source_id, "a prior finding", "shared sub-q")

    crew = ResearchCrew.__new__(ResearchCrew)  # skip __init__
    crew._reuse_checker = _FakeReuseChecker({
        "similarity": 0.91,
        "finding_ids": [finding_id],
        "latency_ms": 2500.0,
        "prompt_tokens": 300,
        "completion_tokens": 120,
        "cost_usd": 0.002,
    })

    monkeypatch.setattr(
        "crew.research_crew.get_researcher",
        lambda: pytest.fail("Researcher must not be called on a full cache hit"),
    )

    new_question_id = repository.create_question("new topic")
    start_run("run-subq-cache")
    set_stage("researcher")
    findings, degraded, unresolved, all_cached = crew._run_research_stage(new_question_id, ["shared sub-q"])
    clear_run()

    assert all_cached is True
    assert degraded is False
    assert unresolved == []
    assert len(findings) == 1
    assert findings[0]["text"] == "a prior finding"
    assert findings[0]["question_id"] == new_question_id

    cache_events = [e for e in _drain("run-subq-cache") if e.event_type == "cache_hit"]
    assert len(cache_events) == 1
    assert cache_events[0].tokens_saved == 420
    assert cache_events[0].cost_avoided_usd == 0.002
    assert cache_events[0].seconds_saved == 2.5


def test_subquestion_cache_miss_falls_back_to_fresh_research_and_indexes_result(monkeypatch):
    crew = ResearchCrew.__new__(ResearchCrew)
    checker = _FakeReuseChecker(None)
    crew._reuse_checker = checker

    class _FakeTask:
        def __init__(self, description):
            self.description = description

    monkeypatch.setattr("crew.research_crew.get_researcher", lambda: object())
    monkeypatch.setattr("crew.research_crew.get_research_task", lambda researcher, sub_question: _FakeTask(sub_question))
    monkeypatch.setattr(
        "crew.research_crew._run_single_task_crew",
        lambda agent, task: '{"findings": [{"finding_text": "fresh finding", "source_title": "t", "source_url": "http://y"}]}',
    )

    question_id = repository.create_question("topic")
    findings, degraded, unresolved, all_cached = crew._run_research_stage(question_id, ["fresh sub-q"])

    assert all_cached is False
    assert degraded is False
    assert len(findings) == 1
    assert len(checker.indexed) == 1
    indexed_sub_question, indexed_finding_ids, *_ = checker.indexed[0]
    assert indexed_sub_question == "fresh sub-q"
    assert indexed_finding_ids == [findings[0]["id"]]
