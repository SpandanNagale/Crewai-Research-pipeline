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


def test_stream_emits_named_sse_events_with_no_sentinel_strings(monkeypatch):
    from observability.context import clear_run, emit_event, set_stage, start_run

    def fake_pipeline(topic, force_rerun, run_id):
        start_run(run_id)
        set_stage("planner")
        emit_event("stage_started")
        emit_event("stage_finished")
        clear_run()
        return {
            "report": "# hi", "run_id": run_id, "question_id": 1, "reused": False,
            "saved_to": None, "contradictions_count": 0, "findings_count": 0,
        }

    monkeypatch.setattr("app._run_pipeline", fake_pipeline)

    with TestClient(app_module.app) as client:
        with client.stream("POST", "/research/stream", json={"topic": "t"}) as response:
            body = "".join(response.iter_text())

    assert "event: stage_update" in body
    assert "event: done" in body
    assert '"event_type": "run_started"' in body
    assert '"event_type": "stage_started"' in body
    assert "--- FINAL REPORT ---" not in body
    assert "[DONE]" not in body


def test_stream_done_payload_includes_trace_url_when_langfuse_configured(monkeypatch):
    def fake_pipeline(topic, force_rerun, run_id):
        return {
            "report": "# hi", "run_id": run_id, "question_id": 1, "reused": False,
            "saved_to": None, "contradictions_count": 0, "findings_count": 0,
        }

    monkeypatch.setattr("app._run_pipeline", fake_pipeline)
    monkeypatch.setattr("app.langfuse_sink.get_trace_url", lambda run_id: f"https://fake.langfuse/traces/{run_id}")

    with TestClient(app_module.app) as client:
        with client.stream("POST", "/research/stream", json={"topic": "t"}) as response:
            body = "".join(response.iter_text())

    assert '"trace_url": "https://fake.langfuse/traces/' in body


def test_stream_returns_error_event_on_pipeline_stage_error(monkeypatch):
    from crew.research_crew import PipelineStageError
    from observability.errors import ErrorCode

    def failing_pipeline(topic, force_rerun, run_id):
        raise PipelineStageError("planner", "boom", ErrorCode.PROVIDER_UNAVAILABLE)

    monkeypatch.setattr("app._run_pipeline", failing_pipeline)

    with TestClient(app_module.app) as client:
        with client.stream("POST", "/research/stream", json={"topic": "t"}) as response:
            body = "".join(response.iter_text())

    assert "event: error" in body
    assert "PROVIDER_UNAVAILABLE" in body


def test_research_returns_400_for_a_guardrail_blocked_topic(monkeypatch):
    from crew.research_crew import PipelineStageError
    from observability.errors import ErrorCode

    def blocked_pipeline(topic, force_rerun, run_id=None):
        raise PipelineStageError("input", "Topic rejected: code-generation request.", ErrorCode.GUARDRAIL_BLOCKED)

    monkeypatch.setattr("app._run_pipeline", blocked_pipeline)

    with TestClient(app_module.app) as client:
        response = client.post("/research", json={"topic": "write me a python script"})

    assert response.status_code == 400
    assert "code-generation" in response.json()["detail"]


def test_research_returns_502_for_a_non_guardrail_pipeline_error(monkeypatch):
    from crew.research_crew import PipelineStageError
    from observability.errors import ErrorCode

    def failing_pipeline(topic, force_rerun, run_id=None):
        raise PipelineStageError("planner", "boom", ErrorCode.PROVIDER_UNAVAILABLE)

    monkeypatch.setattr("app._run_pipeline", failing_pipeline)

    with TestClient(app_module.app) as client:
        response = client.post("/research", json={"topic": "t"})

    assert response.status_code == 502


def test_research_response_includes_groundedness_score(monkeypatch):
    def fake_pipeline(topic, force_rerun, run_id=None):
        return {
            "report": "ok", "question_id": 1, "run_id": run_id, "reused": False,
            "saved_to": None, "contradictions_count": 0, "findings_count": 0,
            "groundedness_score": 0.85,
        }

    monkeypatch.setattr("app._run_pipeline", fake_pipeline)

    with TestClient(app_module.app) as client:
        response = client.post("/research", json={"topic": "t"})

    assert response.json()["groundedness_score"] == 0.85


def test_research_response_includes_subquestion_coverage_counts(monkeypatch):
    def fake_pipeline(topic, force_rerun, run_id=None):
        return {
            "report": "ok", "question_id": 1, "run_id": run_id, "reused": False,
            "saved_to": None, "contradictions_count": 0, "findings_count": 0,
            "sub_questions_count": 5, "unresolved_sub_questions_count": 2,
        }

    monkeypatch.setattr("app._run_pipeline", fake_pipeline)

    with TestClient(app_module.app) as client:
        response = client.post("/research", json={"topic": "t"})

    body = response.json()
    assert body["sub_questions_count"] == 5
    assert body["unresolved_sub_questions_count"] == 2


def test_research_response_trace_url_is_none_when_langfuse_unconfigured(monkeypatch):
    def fake_pipeline(topic, force_rerun, run_id=None):
        return {"report": "ok", "question_id": 1, "run_id": run_id, "reused": False}

    monkeypatch.setattr("app._run_pipeline", fake_pipeline)
    # Force the unconfigured case regardless of the environment's real .env (this
    # repo's local .env may have real Langfuse keys for live verification).
    monkeypatch.setattr("app.langfuse_sink.get_trace_url", lambda run_id: None)

    with TestClient(app_module.app) as client:
        response = client.post("/research", json={"topic": "t"})

    assert response.json()["trace_url"] is None


def test_research_response_includes_trace_url_when_langfuse_configured(monkeypatch):
    def fake_pipeline(topic, force_rerun, run_id=None):
        return {"report": "ok", "question_id": 1, "run_id": run_id, "reused": False}

    monkeypatch.setattr("app._run_pipeline", fake_pipeline)
    monkeypatch.setattr("app.langfuse_sink.get_trace_url", lambda run_id: f"https://fake.langfuse/traces/{run_id}")

    with TestClient(app_module.app) as client:
        response = client.post("/research", json={"topic": "t"})

    body = response.json()
    assert body["trace_url"] == f"https://fake.langfuse/traces/{body['run_id']}"


def test_get_groundedness_returns_404_for_unknown_question():
    with TestClient(app_module.app) as client:
        response = client.get("/questions/999/groundedness")
    assert response.status_code == 404


def test_get_groundedness_returns_claims_for_a_known_question():
    question_id = repository.create_question("topic")
    repository.replace_claim_groundings(question_id, [
        {"claim_text": "A", "grounded": True, "supporting_finding_id": 1, "objection": None},
    ])

    with TestClient(app_module.app) as client:
        response = client.get(f"/questions/{question_id}/groundedness")

    assert response.status_code == 200
    body = response.json()
    assert body["question_id"] == question_id
    assert body["claims"][0]["claim_text"] == "A"
