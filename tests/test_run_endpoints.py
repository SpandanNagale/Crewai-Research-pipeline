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
    app_module._rate_limiter._hits.clear()  # module-level singleton, shared across tests
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

    def fake_pipeline(topic, force_rerun, run_id, *args):
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

    def fake_pipeline(topic, force_rerun, run_id, *args):
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
    def fake_pipeline(topic, force_rerun, run_id, *args):
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

    def failing_pipeline(topic, force_rerun, run_id, *args):
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

    def blocked_pipeline(topic, force_rerun, run_id=None, *args):
        raise PipelineStageError("input", "Topic rejected: code-generation request.", ErrorCode.GUARDRAIL_BLOCKED)

    monkeypatch.setattr("app._run_pipeline", blocked_pipeline)

    with TestClient(app_module.app) as client:
        response = client.post("/research", json={"topic": "write me a python script"})

    assert response.status_code == 400
    assert "code-generation" in response.json()["detail"]


def test_research_returns_502_for_a_non_guardrail_pipeline_error(monkeypatch):
    from crew.research_crew import PipelineStageError
    from observability.errors import ErrorCode

    def failing_pipeline(topic, force_rerun, run_id=None, *args):
        raise PipelineStageError("planner", "boom", ErrorCode.PROVIDER_UNAVAILABLE)

    monkeypatch.setattr("app._run_pipeline", failing_pipeline)

    with TestClient(app_module.app) as client:
        response = client.post("/research", json={"topic": "t"})

    assert response.status_code == 502


def test_research_response_includes_groundedness_score(monkeypatch):
    def fake_pipeline(topic, force_rerun, run_id=None, *args):
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
    def fake_pipeline(topic, force_rerun, run_id=None, *args):
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
    def fake_pipeline(topic, force_rerun, run_id=None, *args):
        return {"report": "ok", "question_id": 1, "run_id": run_id, "reused": False}

    monkeypatch.setattr("app._run_pipeline", fake_pipeline)
    # Force the unconfigured case regardless of the environment's real .env (this
    # repo's local .env may have real Langfuse keys for live verification).
    monkeypatch.setattr("app.langfuse_sink.get_trace_url", lambda run_id: None)

    with TestClient(app_module.app) as client:
        response = client.post("/research", json={"topic": "t"})

    assert response.json()["trace_url"] is None


def test_research_response_includes_trace_url_when_langfuse_configured(monkeypatch):
    def fake_pipeline(topic, force_rerun, run_id=None, *args):
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
    assert body["threshold"] == pytest.approx(0.7)
    assert body["claims"][0]["claim_text"] == "A"


def test_cors_allows_configured_frontend_origin():
    with TestClient(app_module.app) as client:
        response = client.options(
            "/health",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_health_budget_reports_uncapped_by_default(monkeypatch):
    monkeypatch.setattr("app.settings.DAILY_BUDGET_USD", 0)
    with TestClient(app_module.app) as client:
        response = client.get("/health/budget")
    body = response.json()
    assert body["daily_budget_usd"] is None
    assert body["exceeded"] is False
    assert body["remaining_usd"] is None


def test_health_budget_reports_real_spend_against_a_configured_cap(monkeypatch):
    monkeypatch.setattr("app.settings.DAILY_BUDGET_USD", 5.0)
    repository.record_spend(2.0)
    with TestClient(app_module.app) as client:
        response = client.get("/health/budget")
    body = response.json()
    assert body["daily_budget_usd"] == 5.0
    assert body["spent_today_usd"] == pytest.approx(2.0)
    assert body["remaining_usd"] == pytest.approx(3.0)
    assert body["exceeded"] is False


def test_research_returns_503_when_daily_budget_exceeded(monkeypatch):
    monkeypatch.setattr("app.settings.DAILY_BUDGET_USD", 1.0)
    repository.record_spend(1.0)

    def fake_pipeline(topic, force_rerun, run_id=None, *args):
        raise AssertionError("pipeline must not run once the budget is exceeded")

    monkeypatch.setattr("app._run_pipeline", fake_pipeline)

    with TestClient(app_module.app) as client:
        response = client.post("/research", json={"topic": "t"})

    assert response.status_code == 503
    assert "budget" in response.json()["detail"].lower()


def test_research_byok_bypasses_an_exceeded_budget(monkeypatch):
    monkeypatch.setattr("app.settings.DAILY_BUDGET_USD", 1.0)
    repository.record_spend(1.0)
    monkeypatch.setattr("app.find_model_for_provider", lambda provider: "some-model")

    captured = {}

    def fake_pipeline(topic, force_rerun, run_id=None, byok_override=None):
        captured["byok_override"] = byok_override
        return {"report": "ok", "question_id": 1, "run_id": run_id, "reused": False}

    monkeypatch.setattr("app._run_pipeline", fake_pipeline)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/research",
            json={"topic": "t", "byok_provider": "groq", "byok_api_key": "user-key"},
        )

    assert response.status_code == 200
    assert captured["byok_override"].provider == "groq"
    assert captured["byok_override"].api_key == "user-key"
    assert captured["byok_override"].model == "some-model"


def test_research_byok_requires_both_fields():
    with TestClient(app_module.app) as client:
        response = client.post("/research", json={"topic": "t", "byok_provider": "groq"})
    assert response.status_code == 400
    assert "byok_provider and byok_api_key" in response.json()["detail"]


def test_research_byok_rejects_unknown_provider():
    with TestClient(app_module.app) as client:
        response = client.post(
            "/research",
            json={"topic": "t", "byok_provider": "not-a-real-provider", "byok_api_key": "k"},
        )
    assert response.status_code == 400
    assert "Unknown or unconfigured BYOK provider" in response.json()["detail"]


def test_resume_returns_503_when_daily_budget_exceeded_with_no_byok_escape(monkeypatch):
    repository.create_pipeline_run("run-over-budget", question_id=1, topic="t")
    repository.set_run_status("run-over-budget", "failed")
    monkeypatch.setattr("app.settings.DAILY_BUDGET_USD", 1.0)
    repository.record_spend(1.0)

    with TestClient(app_module.app) as client:
        response = client.post("/research/run-over-budget/resume")

    assert response.status_code == 503


def test_rate_limit_blocks_the_nth_plus_one_request_per_ip(monkeypatch):
    monkeypatch.setattr("app.settings.RATE_LIMIT_PER_MINUTE", 2)

    def fake_pipeline(topic, force_rerun, run_id=None, *args):
        return {"report": "ok", "question_id": 1, "run_id": run_id, "reused": False}

    monkeypatch.setattr("app._run_pipeline", fake_pipeline)

    with TestClient(app_module.app) as client:
        r1 = client.post("/research", json={"topic": "t"})
        r2 = client.post("/research", json={"topic": "t"})
        r3 = client.post("/research", json={"topic": "t"})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429


def test_rate_limiter_tracks_ips_independently(monkeypatch):
    """Unit-level (not through TestClient, which doesn't vary request.client.host
    per call): two different client IPs each get their own budget of requests."""
    monkeypatch.setattr("app.settings.RATE_LIMIT_PER_MINUTE", 1)
    limiter = app_module._RateLimiter()

    class _FakeClient:
        def __init__(self, host):
            self.host = host

    class _FakeRequest:
        def __init__(self, host):
            self.client = _FakeClient(host)

    limiter.check(_FakeRequest("1.1.1.1"))
    limiter.check(_FakeRequest("2.2.2.2"))  # different IP, must not be blocked
    with pytest.raises(app_module.HTTPException) as exc_info:
        limiter.check(_FakeRequest("1.1.1.1"))  # same IP again, over its own limit
    assert exc_info.value.status_code == 429


def test_rate_limit_disabled_by_default(monkeypatch):
    def fake_pipeline(topic, force_rerun, run_id=None, *args):
        return {"report": "ok", "question_id": 1, "run_id": run_id, "reused": False}

    monkeypatch.setattr("app._run_pipeline", fake_pipeline)

    with TestClient(app_module.app) as client:
        for _ in range(5):
            response = client.post("/research", json={"topic": "t"})
            assert response.status_code == 200
