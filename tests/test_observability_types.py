from datetime import datetime

from observability.errors import ERROR_SENTENCES, ErrorCode, Severity, sentence_for, severity_for
from observability.events import RunEvent
from observability.node_state import NodeState


def test_run_event_requires_core_fields_and_defaults_optionals_to_none():
    event = RunEvent(
        run_id="r1", seq=1, timestamp=datetime.utcnow(), stage="planner",
        event_type="stage_started",
    )
    assert event.message == ""
    assert event.node_state is None
    assert event.cost_usd is None


def test_run_event_round_trips_through_json():
    event = RunEvent(
        run_id="r1", seq=2, timestamp=datetime.utcnow(), stage="researcher",
        event_type="llm_attempt", model_alias="researcher",
        resolved_deployment="groq:openai/gpt-oss-120b", attempt=1,
        latency_ms=812.4, prompt_tokens=120, completion_tokens=45, cost_usd=0.0,
    )
    restored = RunEvent.model_validate_json(event.model_dump_json())
    assert restored.resolved_deployment == "groq:openai/gpt-oss-120b"
    assert restored.cost_usd == 0.0


def test_every_error_code_has_a_sentence_and_severity():
    for code in ErrorCode:
        assert code in ERROR_SENTENCES
        sentence, severity = ERROR_SENTENCES[code]
        assert isinstance(sentence, str) and sentence
        assert isinstance(severity, Severity)
        assert sentence_for(code) == sentence
        assert severity_for(code) == severity


def test_node_state_values_match_spec():
    expected = {"pending", "running", "succeeded", "degraded", "failed",
                "not_reached", "skipped_cached"}
    assert {s.value for s in NodeState} == expected
