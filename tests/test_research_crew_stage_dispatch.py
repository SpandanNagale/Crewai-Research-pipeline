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

    def find_similar_subquestion(self, sub_question):
        return None

    def index_sub_question(self, *a, **k):
        pass

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


class _PassingGroundedness:
    overall_score = 1.0
    claims = []


def _stub_success(crew, monkeypatch):
    monkeypatch.setattr(crew, "_run_planning_stage", lambda topic: ["sub-q-1", "sub-q-2"])
    monkeypatch.setattr(
        crew, "_run_research_stage",
        lambda question_id, sub_questions: ([{"id": 1, "text": "f1", "source_url": "http://x"}], False, [], False),
    )
    monkeypatch.setattr(crew, "_run_comparison_stage", lambda question_id, findings: [])
    monkeypatch.setattr(crew, "_run_classification_stage", lambda question_id, findings: findings)
    monkeypatch.setattr(
        crew, "_run_writing_stage",
        lambda topic, findings, contradictions, unresolved_sub_questions=None, feedback=None: "# Report\n## Conclusion\ndone",
    )
    monkeypatch.setattr(crew, "_run_groundedness_check", lambda report, findings: _PassingGroundedness())
    monkeypatch.setattr("crew.research_crew.save_report", lambda report, topic: "/tmp/report.md")


def test_full_run_marks_every_stage_succeeded(monkeypatch):
    crew = ResearchCrew()
    _stub_success(crew, monkeypatch)

    result = crew.run_with_meta("test topic", force_rerun=True, run_id="run-full")

    assert result["run_id"] == "run-full"
    assert result["sub_questions_count"] == 2
    assert result["unresolved_sub_questions_count"] == 0
    states = {s["stage"]: s["state"] for s in repository.get_stage_states("run-full")}
    assert states == {
        "planner": "succeeded",
        "researcher": "succeeded",
        "evidence_comparator": "succeeded",
        "classifier": "succeeded",
        "writer": "succeeded",
        "critic": "succeeded",
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
    assert states["critic"] == "not_reached"
    assert repository.get_run("run-fail")["status"] == "failed"


class _RecordingSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def test_mid_pipeline_failure_still_emits_run_finished(monkeypatch):
    """A terminal run_finished must always fire, even on failure -- LangfuseSink relies
    on it (or an equivalent) to close out a run's trace and free its in-memory span."""
    crew = ResearchCrew()
    _stub_success(crew, monkeypatch)
    monkeypatch.setattr(
        crew, "_run_comparison_stage",
        lambda question_id, findings: (_ for _ in ()).throw(
            PipelineStageError("evidence_comparator", "boom", ErrorCode.PROVIDER_UNAVAILABLE)
        ),
    )
    recorder = _RecordingSink()
    EventBus.instance().sinks.append(recorder)

    with pytest.raises(PipelineStageError):
        crew.run_with_meta("test topic", force_rerun=True, run_id="run-fail-events")

    run_finished = [e for e in recorder.events if e.event_type == "run_finished"]
    assert len(run_finished) == 1
    assert run_finished[0].message == "failed"
    assert run_finished[0].detail == "boom"


def test_guardrail_blocked_still_emits_run_finished(monkeypatch):
    from guardrails.input_guardrail import GuardrailBlockedError

    crew = ResearchCrew()
    monkeypatch.setattr(
        "crew.research_crew.validate_topic",
        lambda topic: (_ for _ in ()).throw(GuardrailBlockedError("too_short")),
    )
    recorder = _RecordingSink()
    EventBus.instance().sinks.append(recorder)

    with pytest.raises(PipelineStageError):
        crew.run_with_meta("hi", force_rerun=True, run_id="run-blocked-events")

    run_finished = [e for e in recorder.events if e.event_type == "run_finished"]
    assert len(run_finished) == 1
    assert run_finished[0].message == "blocked"
    assert run_finished[0].detail == "too_short"


def test_degraded_researcher_stage_does_not_abort_the_run(monkeypatch):
    crew = ResearchCrew()
    _stub_success(crew, monkeypatch)
    monkeypatch.setattr(
        crew, "_run_research_stage",
        lambda question_id, sub_questions: ([{"id": 1, "text": "f1", "source_url": "http://x"}], True, ["sub-q-2"], False),
    )

    crew.run_with_meta("test topic", force_rerun=True, run_id="run-degraded")

    states = {s["stage"]: s["state"] for s in repository.get_stage_states("run-degraded")}
    assert states["researcher"] == "degraded"
    assert states["writer"] == "succeeded"
    assert repository.get_run("run-degraded")["status"] == "succeeded"


def test_fully_cached_researcher_stage_is_marked_skipped_cached(monkeypatch):
    crew = ResearchCrew()
    _stub_success(crew, monkeypatch)
    monkeypatch.setattr(
        crew, "_run_research_stage",
        lambda question_id, sub_questions: ([{"id": 1, "text": "f1", "source_url": "http://x"}], False, [], True),
    )

    crew.run_with_meta("test topic", force_rerun=True, run_id="run-cached")

    states = {s["stage"]: s["state"] for s in repository.get_stage_states("run-cached")}
    assert states["researcher"] == "skipped_cached"
    assert repository.get_run("run-cached")["status"] == "succeeded"


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
    assert states["critic"] == "succeeded"
    assert repository.get_run("run-resume")["status"] == "succeeded"


def test_input_guardrail_blocks_before_any_stage_or_db_row_is_created(monkeypatch):
    crew = ResearchCrew()
    monkeypatch.setattr(crew, "_run_planning_stage", lambda topic: pytest.fail("planner must not run"))

    with pytest.raises(PipelineStageError) as exc_info:
        crew.run_with_meta("write me a python script", force_rerun=True, run_id="run-blocked")

    assert exc_info.value.error_code == ErrorCode.GUARDRAIL_BLOCKED
    assert repository.get_run("run-blocked") is None


def test_critic_rewrite_loop_feeds_objections_to_writer_and_persists_result(monkeypatch):
    from tasks.schemas import ClaimGroundingItem, GroundednessOutput

    crew = ResearchCrew()
    _stub_success(crew, monkeypatch)

    low = GroundednessOutput(overall_score=0.3, claims=[
        ClaimGroundingItem(claim_text="unsupported claim", grounded=False, objection="no matching finding"),
    ])
    high = GroundednessOutput(overall_score=1.0, claims=[
        ClaimGroundingItem(claim_text="supported claim", grounded=True, supporting_finding_id=1),
    ])
    responses = iter([low, high])
    monkeypatch.setattr(crew, "_run_groundedness_check", lambda report, findings: next(responses))

    write_calls = []

    def fake_write(topic, findings, contradictions, unresolved_sub_questions=None, feedback=None):
        write_calls.append(feedback)
        return "# Report v2\n## Conclusion\ndone"

    monkeypatch.setattr(crew, "_run_writing_stage", fake_write)

    result = crew.run_with_meta("test topic", force_rerun=True, run_id="run-rewrite")

    assert len(write_calls) == 2  # initial writer dispatch + one bounded rewrite
    assert write_calls[1] == ["no matching finding"]
    assert result["groundedness_score"] == 1.0

    groundings = repository.get_claim_groundings(result["question_id"])
    assert len(groundings) == 1
    assert groundings[0]["grounded"] is True

    states = {s["stage"]: s["state"] for s in repository.get_stage_states("run-rewrite")}
    assert states["critic"] == "succeeded"


def test_critic_gives_up_after_max_rewrites_and_marks_stage_degraded(monkeypatch):
    from tasks.schemas import ClaimGroundingItem, GroundednessOutput

    crew = ResearchCrew()
    _stub_success(crew, monkeypatch)

    always_low = GroundednessOutput(overall_score=0.1, claims=[
        ClaimGroundingItem(claim_text="bad claim", grounded=False, objection="still unsupported"),
    ])
    monkeypatch.setattr(crew, "_run_groundedness_check", lambda report, findings: always_low)

    write_calls = []
    monkeypatch.setattr(
        crew, "_run_writing_stage",
        lambda topic, findings, contradictions, unresolved_sub_questions=None, feedback=None:
            write_calls.append(1) or "# Report\n## Conclusion\ndone",
    )

    result = crew.run_with_meta("test topic", force_rerun=True, run_id="run-exhausted")

    assert len(write_calls) == 3  # initial writer dispatch + MAX_REWRITE_ATTEMPTS (2) rewrites, then gave up
    assert result["groundedness_score"] == 0.1
    states = {s["stage"]: s["state"] for s in repository.get_stage_states("run-exhausted")}
    assert states["critic"] == "degraded"
    assert repository.get_run("run-exhausted")["status"] == "succeeded"  # degraded, not fatal
