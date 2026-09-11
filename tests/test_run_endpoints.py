import time
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app as app_module
import db.session as db_session
from db import repository
from db.models import Base


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    # StaticPool (not just check_same_thread=False): TestClient runs endpoint
    # handling -- and this app's loop.run_in_executor() calls -- on threads
    # other than pytest's, and sqlite's default SingletonThreadPool hands each
    # thread its own separate (and separately empty) :memory: database.
    # StaticPool keeps every thread on the one real connection.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    test_session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_session, "SessionLocal", test_session_local)
    yield


def test_get_run_returns_404_for_unknown_run_id():
    with TestClient(app_module.app) as client:
        response = client.get(f"/research/{uuid.uuid4()}")
    assert response.status_code == 404


def test_get_run_returns_run_and_stages():
    repository.create_pipeline_run("run-x", question_id=None, topic="t")
    repository.set_stage_state("run-x", "planner", "succeeded")

    with TestClient(app_module.app) as client:
        response = client.get("/research/run-x")

    body = response.json()
    assert response.status_code == 200
    assert body["run_id"] == "run-x"
    assert body["stages"][0]["stage"] == "planner"
    assert body["stages"][0]["state"] == "succeeded"


def test_resume_returns_404_for_unknown_run_id():
    with TestClient(app_module.app) as client:
        response = client.post(f"/research/{uuid.uuid4()}/resume")
    assert response.status_code == 404


def test_resume_returns_409_for_an_already_succeeded_run():
    repository.create_pipeline_run("run-done", question_id=1, topic="t")
    repository.set_run_status("run-done", "succeeded")

    with TestClient(app_module.app) as client:
        response = client.post("/research/run-done/resume")
    assert response.status_code == 409


def test_resume_re_enters_the_pipeline_with_the_same_run_id(monkeypatch):
    repository.create_pipeline_run("run-retry", question_id=7, topic="my topic")
    repository.set_run_status("run-retry", "failed")

    captured = {}

    def fake_run_with_meta(self, topic, force_rerun=False, run_id=None, resume_question_id=None):
        captured.update(topic=topic, force_rerun=force_rerun, run_id=run_id, resume_question_id=resume_question_id)
        return {
            "report": "ok", "question_id": 7, "run_id": run_id, "reused": False,
            "saved_to": None, "contradictions_count": 0, "findings_count": 0,
        }

    monkeypatch.setattr("app.ResearchCrew.run_with_meta", fake_run_with_meta)

    with TestClient(app_module.app) as client:
        response = client.post("/research/run-retry/resume")

    assert response.status_code == 200
    assert captured == {
        "topic": "my topic", "force_rerun": False, "run_id": "run-retry", "resume_question_id": 7,
    }


def test_async_job_id_matches_the_pipelines_run_id(monkeypatch):
    seen_run_ids = []

    def fake_pipeline(topic, force_rerun, run_id):
        seen_run_ids.append(run_id)
        return {"report": "ok", "question_id": 1, "run_id": run_id}

    monkeypatch.setattr("app._run_pipeline", fake_pipeline)

    with TestClient(app_module.app) as client:
        response = client.post("/research/async", json={"topic": "t"})
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        for _ in range(50):
            job = client.get(f"/jobs/{job_id}").json()
            if job["status"] == "done":
                break
            time.sleep(0.05)

    assert seen_run_ids == [job_id]
