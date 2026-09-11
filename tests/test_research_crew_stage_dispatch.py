import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import db.session as db_session
from crew.research_crew import PipelineStageError, ResearchCrew
from db import repository
from db.models import Base
from observability.bus import EventBus
from observability.errors import ErrorCode


class _DummyReuseChecker:
    def find_similar(self, topic):
        return None

    def index_question(self, *a, **k):
        pass

    def index_conclusion(self, *a, **k):
        pass

    def index_finding(self, *a, **k):
        pass


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    test_session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_session, "SessionLocal", test_session_local)
    monkeypatch.setattr("crew.research_crew.init_db", lambda: None)
    monkeypatch.setattr("crew.research_crew.KnowledgeReuseChecker", _DummyReuseChecker)
    EventBus.reset_for_tests()
    yield


def _stub_success(crew, monkeypatch):
    monkeypatch.setattr(crew, "_run_planning_stage", lambda topic: ["sub-q-1", "sub-q-2"])
    monkeypatch.setattr(
        crew, "_run_research_stage",
        lambda question_id, sub_questions: ([{"id": 1, "text": "f1", "source_url": "http://x"}], False),
    )
    monkeypatch.setattr(crew, "_run_comparison_stage", lambda question_id, findings: [])
    monkeypatch.setattr(crew, "_run_classification_stage", lambda question_id, findings: findings)
    monkeypatch.setattr(
        crew, "_run_writing_stage",
        lambda topic, findings, contradictions: "# Report\n## Conclusion\ndone",
    )
    monkeypatch.setattr("crew.research_crew.save_report", lambda report, topic: "/tmp/report.md")


def test_full_run_marks_every_stage_succeeded(monkeypatch):
    crew = ResearchCrew()
    _stub_success(crew, monkeypatch)

    result = crew.run_with_meta("test topic", force_rerun=True, run_id="run-full")

    assert result["run_id"] == "run-full"
    states = {s["stage"]: s["state"] for s in repository.get_stage_states("run-full")}
    assert states == {
        "planner": "succeeded",
        "researcher": "succeeded",
        "evidence_comparator": "succeeded",
        "classifier": "succeeded",
        "writer": "succeeded",
    }
    assert repository.get_run("run-full")["status"] == "succeeded"


def test_mid_pipeline_failure_marks_downstream_stages_not_reached(monkeypatch):
    crew = ResearchCrew()
    _stub_success(crew, monkeypatch)
    monkeypatch.setattr(
        crew, "_run_comparison_stage",
        lambda question_id, findings: (_ for _ in ()).throw(
            PipelineStageError("evidence_comparator", "boom", ErrorCode.PROVIDER_UNAVAILABLE)
        ),
    )

    with pytest.raises(PipelineStageError):
        crew.run_with_meta("test topic", force_rerun=True, run_id="run-fail")

    states = {s["stage"]: s["state"] for s in repository.get_stage_states("run-fail")}
    assert states["planner"] == "succeeded"
    assert states["researcher"] == "succeeded"
    assert states["evidence_comparator"] == "failed"
    assert states["classifier"] == "not_reached"
    assert states["writer"] == "not_reached"
    assert repository.get_run("run-fail")["status"] == "failed"


def test_degraded_researcher_stage_does_not_abort_the_run(monkeypatch):
    crew = ResearchCrew()
    _stub_success(crew, monkeypatch)
    monkeypatch.setattr(
        crew, "_run_research_stage",
        lambda question_id, sub_questions: ([{"id": 1, "text": "f1", "source_url": "none"}], True),
    )

    crew.run_with_meta("test topic", force_rerun=True, run_id="run-degraded")

    states = {s["stage"]: s["state"] for s in repository.get_stage_states("run-degraded")}
    assert states["researcher"] == "degraded"
    assert states["writer"] == "succeeded"
    assert repository.get_run("run-degraded")["status"] == "succeeded"


def test_resume_skips_already_succeeded_stages_and_does_not_rerun_the_planner(monkeypatch):
    crew = ResearchCrew()
    _stub_success(crew, monkeypatch)

    planning_calls = []
    real_planning = crew._run_planning_stage

    def _counting_planning(topic):
        planning_calls.append(topic)
        return real_planning(topic)

    monkeypatch.setattr(crew, "_run_planning_stage", _counting_planning)
    monkeypatch.setattr(
        crew, "_run_comparison_stage",
        lambda question_id, findings: (_ for _ in ()).throw(
            PipelineStageError("evidence_comparator", "boom", ErrorCode.PROVIDER_UNAVAILABLE)
        ),
    )

    with pytest.raises(PipelineStageError):
        crew.run_with_meta("test topic", force_rerun=True, run_id="run-resume")

    question_id = repository.get_run("run-resume")["question_id"]
    assert len(planning_calls) == 1

    monkeypatch.setattr(crew, "_run_comparison_stage", lambda question_id, findings: [])
    crew.run_with_meta("test topic", run_id="run-resume", resume_question_id=question_id)

    assert len(planning_calls) == 1  # planner was not re-run on resume
    states = {s["stage"]: s["state"] for s in repository.get_stage_states("run-resume")}
    assert states["planner"] == "skipped_cached"
    assert states["researcher"] == "skipped_cached"
    assert states["evidence_comparator"] == "succeeded"
    assert states["writer"] == "succeeded"
    assert repository.get_run("run-resume")["status"] == "succeeded"
