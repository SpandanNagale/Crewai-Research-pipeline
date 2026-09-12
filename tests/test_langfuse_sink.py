from datetime import datetime

import pytest
from opentelemetry import trace as otel_trace_api

from config.settings import settings
from observability import langfuse_sink
from observability.events import RunEvent
from observability.node_state import NodeState


_next_fake_span_id = [1]


def _make_fake_otel_span():
    """otel_trace_api.get_current_span() only ever returns real Span instances
    (see its source: `isinstance(span, Span)` or it falls back to INVALID_SPAN) --
    so the fake parent otel span used to test langfuse_sink._start_nested's real
    OTel-context nesting must actually be one, not a bare duck-typed object.
    NonRecordingSpan is the SDK's own minimal concrete Span."""
    span_id = _next_fake_span_id[0]
    _next_fake_span_id[0] += 1
    span = otel_trace_api.NonRecordingSpan(otel_trace_api.SpanContext(
        trace_id=1, span_id=span_id, is_remote=False,
        trace_flags=otel_trace_api.TraceFlags(0x01),
    ))
    span._id = span_id  # convenience for test assertions below
    return span


def _current_otel_span_id():
    """The id() of whatever FakeOtelSpan is ambient via otel_trace_api.use_span(),
    or None if there is no real ambient span (e.g. the root, created via the
    explicit trace_context={"parent_span_id": None} path)."""
    ctx = otel_trace_api.get_current_span().get_span_context()
    return ctx.span_id if ctx.is_valid else None


class FakeObservation:
    def __init__(self, name, as_type, **kwargs):
        self.name = name
        self.as_type = as_type
        self.kwargs = kwargs
        self.id = f"span-{id(self):x}"[:16]
        self.trace_id = kwargs.get("trace_context", {}).get("trace_id")
        # Nested children are created under otel_trace_api.use_span(parent._otel_span, ...)
        # (real OTel context propagation, not trace_context) -- capture that here so
        # tests can assert "child of X" without the sink passing trace_context at all.
        self.parent_from_context = _current_otel_span_id()
        self._otel_span = _make_fake_otel_span()
        self.ended = False
        self.end_calls = []
        self.update_calls = []

    def update(self, **kwargs):
        self.update_calls.append(kwargs)
        return self

    def end(self, **kwargs):
        self.ended = True
        self.end_calls.append(kwargs)
        return self


class FakeLangfuseClient:
    def __init__(self):
        self.observations = []
        self.events = []
        self.scores = []
        self.flush_count = 0

    def start_observation(self, *, name, as_type="span", **kwargs):
        obs = FakeObservation(name, as_type, **kwargs)
        self.observations.append(obs)
        return obs

    def create_event(self, *, name, **kwargs):
        self.events.append({"name": name, "parent_from_context": _current_otel_span_id(), **kwargs})

    def create_score(self, *, trace_id, name, value, **kwargs):
        self.scores.append({"trace_id": trace_id, "name": name, "value": value, **kwargs})

    def flush(self):
        self.flush_count += 1

    def get_trace_url(self, *, trace_id):
        return f"https://fake.langfuse/project/p1/traces/{trace_id}"


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeLangfuseClient()
    monkeypatch.setattr(langfuse_sink, "_get_client", lambda: client)
    return client


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "sk-test")


def _event(**overrides) -> RunEvent:
    fields = dict(run_id="run-1", seq=1, timestamp=datetime.utcnow(), stage="planner", event_type="stage_started")
    fields.update(overrides)
    return RunEvent(**fields)


def test_is_configured_false_without_keys(monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "")
    assert langfuse_sink.is_configured() is False


def test_is_configured_true_with_both_keys(configured):
    assert langfuse_sink.is_configured() is True


def test_trace_id_for_is_deterministic_and_valid_hex():
    a = langfuse_sink.trace_id_for("run-1")
    b = langfuse_sink.trace_id_for("run-1")
    c = langfuse_sink.trace_id_for("run-2")
    assert a == b
    assert a != c
    assert len(a) == 32
    int(a, 16)  # valid hex


def test_get_trace_url_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "")
    assert langfuse_sink.get_trace_url("run-1") is None


def test_get_trace_url_delegates_to_client(configured, fake_client):
    url = langfuse_sink.get_trace_url("run-1")
    assert url == f"https://fake.langfuse/project/p1/traces/{langfuse_sink.trace_id_for('run-1')}"


def test_get_trace_url_swallows_client_errors(configured, monkeypatch):
    def boom():
        raise RuntimeError("network down")
    monkeypatch.setattr(langfuse_sink, "_get_client", boom)
    assert langfuse_sink.get_trace_url("run-1") is None


def test_attach_score_noop_when_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "")
    called = []
    monkeypatch.setattr(langfuse_sink, "_get_client", lambda: called.append(True))
    langfuse_sink.attach_score("run-1", "groundedness", 0.9)
    assert called == []


def test_attach_score_records_numeric_score_and_flushes(configured, fake_client):
    langfuse_sink.attach_score("run-1", "citation_precision", 0.75, comment="from harness")
    assert fake_client.scores == [{
        "trace_id": langfuse_sink.trace_id_for("run-1"),
        "name": "citation_precision", "value": 0.75,
        "data_type": "NUMERIC", "comment": "from harness",
    }]
    assert fake_client.flush_count == 1


def test_attach_score_swallows_client_errors(configured, monkeypatch):
    monkeypatch.setattr(langfuse_sink, "_get_client", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    langfuse_sink.attach_score("run-1", "groundedness", 0.5)  # must not raise


class TestLangfuseSinkLifecycle:
    def test_run_started_creates_root_span_with_no_parent(self, fake_client):
        sink = langfuse_sink.LangfuseSink()
        sink.emit(_event(event_type="run_started", stage="", message="topic='x'"))

        assert len(fake_client.observations) == 1
        root = fake_client.observations[0]
        assert root.as_type == "span"
        assert root.kwargs["trace_context"]["parent_span_id"] is None
        assert root.kwargs["trace_context"]["trace_id"] == langfuse_sink.trace_id_for("run-1")

    def test_stage_started_nests_under_root(self, fake_client):
        sink = langfuse_sink.LangfuseSink()
        sink.emit(_event(event_type="run_started", stage=""))
        root = fake_client.observations[0]

        sink.emit(_event(event_type="stage_started", stage="planner"))
        stage_span = fake_client.observations[1]
        assert stage_span.name == "planner"
        assert stage_span.parent_from_context == root._otel_span._id

    def test_llm_attempt_creates_generation_under_current_stage(self, fake_client):
        sink = langfuse_sink.LangfuseSink()
        sink.emit(_event(event_type="run_started", stage=""))
        sink.emit(_event(event_type="stage_started", stage="researcher"))
        stage_span = fake_client.observations[1]

        sink.emit(_event(
            event_type="llm_attempt", stage="researcher", model_alias="researcher",
            resolved_deployment="groq:llama", attempt=0, latency_ms=120.0,
            prompt_tokens=100, completion_tokens=50, cost_usd=0.002, message="ok",
        ))

        gen = fake_client.observations[-1]
        assert gen.as_type == "generation"
        assert gen.parent_from_context == stage_span._otel_span._id
        assert gen.kwargs["model"] == "groq:llama"
        assert gen.kwargs["usage_details"] == {"input": 100, "output": 50}
        assert gen.kwargs["cost_details"] == {"total": 0.002}
        assert gen.ended is True

    def test_llm_attempt_failure_marked_warning_level(self, fake_client):
        sink = langfuse_sink.LangfuseSink()
        sink.emit(_event(event_type="run_started", stage=""))
        sink.emit(_event(event_type="stage_started", stage="researcher"))
        sink.emit(_event(
            event_type="llm_attempt", stage="researcher", attempt=1,
            message="provider_timeout", detail="boom",
        ))
        gen = fake_client.observations[-1]
        assert gen.kwargs["level"] == "WARNING"

    def test_cache_hit_emitted_as_event_child_of_current_stage(self, fake_client):
        sink = langfuse_sink.LangfuseSink()
        sink.emit(_event(event_type="run_started", stage=""))
        sink.emit(_event(event_type="stage_started", stage="researcher"))
        stage_span = fake_client.observations[1]

        sink.emit(_event(event_type="cache_hit", stage="researcher", message="search cache hit", cache_hit=True))

        assert len(fake_client.events) == 1
        assert fake_client.events[0]["name"] == "cache_hit"
        assert fake_client.events[0]["parent_from_context"] == stage_span._otel_span._id

    def test_guardrail_fired_uses_guardrail_observation_type(self, fake_client):
        sink = langfuse_sink.LangfuseSink()
        sink.emit(_event(event_type="run_started", stage=""))
        sink.emit(_event(event_type="guardrail_fired", stage="", message="blocked", detail="too_short"))

        guardrail_obs = [o for o in fake_client.observations if o.as_type == "guardrail"]
        assert len(guardrail_obs) == 1
        assert guardrail_obs[0].ended is True

    def test_stage_finished_ends_span_and_records_node_state(self, fake_client):
        sink = langfuse_sink.LangfuseSink()
        sink.emit(_event(event_type="run_started", stage=""))
        sink.emit(_event(event_type="stage_started", stage="planner"))
        stage_span = fake_client.observations[1]

        sink.emit(_event(event_type="stage_finished", stage="planner", node_state=NodeState.SUCCEEDED))

        assert stage_span.ended is True
        assert stage_span.update_calls[-1]["output"] == {"node_state": "succeeded"}

    def test_stage_failed_marks_span_error_without_ending_it(self, fake_client):
        sink = langfuse_sink.LangfuseSink()
        sink.emit(_event(event_type="run_started", stage=""))
        sink.emit(_event(event_type="stage_started", stage="planner"))
        stage_span = fake_client.observations[1]

        sink.emit(_event(event_type="stage_failed", stage="planner", detail="LLM exploded"))

        assert stage_span.update_calls[-1]["level"] == "ERROR"
        assert stage_span.update_calls[-1]["status_message"] == "LLM exploded"
        assert stage_span.ended is False

    def test_run_finished_ends_root_and_clears_run_state(self, fake_client):
        sink = langfuse_sink.LangfuseSink()
        sink.emit(_event(event_type="run_started", stage=""))
        root = fake_client.observations[0]

        sink.emit(_event(event_type="run_finished", stage="", message="ok"))

        assert root.ended is True
        assert sink._runs == {}
        assert fake_client.flush_count == 1

    def test_run_finished_with_score_attaches_it_to_the_trace(self, configured, fake_client):
        sink = langfuse_sink.LangfuseSink()
        sink.emit(_event(event_type="run_started", stage=""))
        sink.emit(_event(
            event_type="run_finished", stage="", message="ok",
            score_name="groundedness", score_value=0.83,
        ))

        assert fake_client.scores == [{
            "trace_id": langfuse_sink.trace_id_for("run-1"),
            "name": "groundedness", "value": 0.83,
            "data_type": "NUMERIC", "comment": None,
        }]

    def test_events_before_run_started_are_dropped_safely(self, fake_client):
        sink = langfuse_sink.LangfuseSink()
        sink.emit(_event(event_type="stage_started", stage="planner"))  # no run_started yet
        assert fake_client.observations == []

    def test_emit_never_raises_even_if_client_construction_fails(self, monkeypatch):
        sink = langfuse_sink.LangfuseSink()
        monkeypatch.setattr(langfuse_sink, "_get_client", lambda: (_ for _ in ()).throw(RuntimeError("down")))
        sink.emit(_event(event_type="run_started", stage=""))  # must not raise
        assert sink._runs == {}


def test_event_bus_adds_langfuse_sink_only_when_configured(monkeypatch):
    from observability.bus import EventBus, NoOpTraceSink

    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "")
    bus = EventBus()
    assert not any(isinstance(s, langfuse_sink.LangfuseSink) for s in bus.sinks)
    assert any(isinstance(s, NoOpTraceSink) for s in bus.sinks)

    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "sk-test")
    bus2 = EventBus()
    assert any(isinstance(s, langfuse_sink.LangfuseSink) for s in bus2.sinks)
