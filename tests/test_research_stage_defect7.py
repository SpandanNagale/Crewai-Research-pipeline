import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import db.session as db_session
from crew.research_crew import ResearchCrew
from db import repository
from db.models import Base


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    test_session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_session, "SessionLocal", test_session_local)
    yield


def test_unresolved_sub_question_never_becomes_a_placeholder_finding(monkeypatch):
    """
    Defect #7 regression: a sub-question that fails to parse after retrying
    must be reported as unresolved, never inserted as a Finding with source
    "none" that would silently flow into the Writer as if it were real content.
    """
    question_id = repository.create_question("topic")

    class _FakeTask:
        def __init__(self, description):
            self.description = description

    def fake_run_single_task_crew(agent, task):
        if task.description == "good-q":
            return '{"findings": [{"finding_text": "x", "source_title": "t", "source_url": "http://x"}]}'
        raise RuntimeError("malformed output")

    monkeypatch.setattr("crew.research_crew.get_researcher", lambda: object())
    monkeypatch.setattr("crew.research_crew.get_research_task", lambda researcher, sub_question: _FakeTask(sub_question))
    monkeypatch.setattr("crew.research_crew._run_single_task_crew", fake_run_single_task_crew)

    crew = ResearchCrew.__new__(ResearchCrew)  # skip __init__ (no DB/reuse-checker needed here)
    findings, degraded, unresolved = crew._run_research_stage(question_id, ["good-q", "bad-q"])

    assert degraded is True
    assert unresolved == ["bad-q"]
    assert len(findings) == 1
    assert all(f["sub_question"] != "bad-q" for f in repository.get_findings_list(question_id))
    assert all(f["source_url"] != "none" for f in repository.get_findings_list(question_id))
