import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import db.session as db_session
from db import repository
from db.models import Base


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    test_session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_session, "SessionLocal", test_session_local)
    yield


def test_create_pipeline_run_and_get_run_round_trip():
    repository.create_pipeline_run("run-1", question_id=None, topic="quantum computing")
    run = repository.get_run("run-1")
    assert run["run_id"] == "run-1"
    assert run["question_id"] is None
    assert run["topic"] == "quantum computing"
    assert run["status"] == "running"


def test_set_stage_state_upserts_the_same_row_across_transitions():
    repository.create_pipeline_run("run-2", question_id=None, topic="x")
    repository.set_stage_state("run-2", "planner", "running")
    repository.set_stage_state("run-2", "planner", "succeeded", attempts=[{"attempt": 0, "message": "ok"}])

    states = repository.get_stage_states("run-2")
    assert len(states) == 1
    assert states[0]["stage"] == "planner"
    assert states[0]["state"] == "succeeded"
    assert states[0]["attempts"] == [{"attempt": 0, "message": "ok"}]
    assert states[0]["started_at"] is not None
    assert states[0]["finished_at"] is not None


def test_set_stage_state_tracks_multiple_stages_independently():
    repository.create_pipeline_run("run-3", question_id=None, topic="x")
    repository.set_stage_state("run-3", "planner", "succeeded")
    repository.set_stage_state("run-3", "researcher", "failed", detail="provider unavailable")

    states = {s["stage"]: s for s in repository.get_stage_states("run-3")}
    assert states["planner"]["state"] == "succeeded"
    assert states["researcher"]["state"] == "failed"
    assert states["researcher"]["detail"] == "provider unavailable"


def test_set_run_status_updates_existing_run():
    repository.create_pipeline_run("run-4", question_id=None, topic="x")
    repository.set_run_status("run-4", "failed")
    assert repository.get_run("run-4")["status"] == "failed"


def test_reattach_pipeline_run_links_question_id_for_resume():
    repository.create_pipeline_run("run-5", question_id=None, topic="x")
    repository.reattach_pipeline_run("run-5", question_id=42)
    run = repository.get_run("run-5")
    assert run["question_id"] == 42
    assert run["status"] == "running"


def test_get_run_returns_none_for_unknown_run_id():
    assert repository.get_run("nope") is None
