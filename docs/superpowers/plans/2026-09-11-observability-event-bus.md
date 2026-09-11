# Phase 2 — Observability spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the pipeline a `RunEvent` stream, a closed `ErrorCode` taxonomy, an
explicit per-stage `NodeState` machine, and a resume path — the substrate Phase 3
(real streaming) and Phase 8 (UI) render on top of.

**Architecture:** A new `observability/` package holds pure types (`RunEvent`,
`ErrorCode`, `NodeState`), a fan-out `EventBus` (stdout JSON sink + in-memory
per-run queue sink for a future SSE consumer + a no-op Langfuse-shaped stub), and a
`threading.local`-based `RunContext` that lets any code running on a pipeline's worker
thread call `emit_event(...)` without a parameter threaded through every call site.
`crew/research_crew.py`'s `run_with_meta` is refactored from a hardcoded chain of
stage calls into a stage-dispatch loop over `STAGES`, persisting per-stage `NodeState`
to two new additive DB tables and enabling `POST /research/{run_id}/resume`.

**Tech Stack:** Python, FastAPI, SQLAlchemy (sqlite), Pydantic, pytest. No new
third-party dependency — `threading`/`queue`/`uuid` are stdlib.

**Spec:** `docs/superpowers/specs/2026-09-11-observability-event-bus-design.md`

## Global Constraints

- `get_llm(agent_name)` stays the only interface agent code calls. No file under
  `agents/` or `tasks/` changes signature or gains a new parameter.
- `crew/research_crew.py`'s existing stage bodies (`_run_planning_stage`,
  `_run_research_stage`, etc.) and `_extract_pydantic`'s raw-JSON parsing keep their
  internal logic — they get wrapped by a dispatch loop, not rewritten.
- `PipelineStageError(stage, detail)` stays backward compatible: `error_code` is a
  third, defaulted parameter.
- A sink raising inside `EventBus.emit()` must never propagate into the pipeline.
- Never fabricate a `cost_usd` — an unpriced (provider, model) pair returns `None`.
- No changes to the `research_questions`/`sources`/`findings`/`contradictions`/
  `conclusions` tables. New state lives in two additive tables only.
- This pass does not rewrite `/research/stream` (Phase 3) and does not implement a
  real Langfuse sink (Phase 7) — both get a stub this plan's plumbing already reaches.

---

### Task 1: Core observability types — `RunEvent`, `ErrorCode`, `NodeState`

**Files:**
- Create: `observability/__init__.py` (empty)
- Create: `observability/node_state.py`
- Create: `observability/errors.py`
- Create: `observability/events.py`
- Test: `tests/test_observability_types.py`

**Interfaces:**
- Produces: `observability.node_state.NodeState` (str enum: `pending, running,
  succeeded, degraded, failed, not_reached, skipped_cached`). `observability.errors.
  Severity` (str enum: `info, warning, error`), `observability.errors.ErrorCode` (str
  enum with 11 members per the spec's §5.2 list), `observability.errors.
  ERROR_SENTENCES: dict[ErrorCode, tuple[str, Severity]]`, `observability.errors.
  sentence_for(code) -> str`, `observability.errors.severity_for(code) -> Severity`.
  `observability.events.RunEvent` (pydantic `BaseModel`) with fields: `run_id: str,
  seq: int, timestamp: datetime, stage: str, event_type: str, node_state:
  NodeState | None = None, model_alias: str | None = None, resolved_deployment:
  str | None = None, attempt: int | None = None, latency_ms: float | None = None,
  prompt_tokens: int | None = None, completion_tokens: int | None = None, cost_usd:
  float | None = None, cache_hit: bool | None = None, message: str = "", detail:
  str | None = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_observability_types.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_observability_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'observability'`

- [ ] **Step 3: Write the implementation**

```python
# observability/node_state.py
from enum import Enum


class NodeState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    DEGRADED = "degraded"
    FAILED = "failed"
    NOT_REACHED = "not_reached"
    SKIPPED_CACHED = "skipped_cached"
```

```python
# observability/errors.py
from enum import Enum
from typing import Dict, Tuple


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ErrorCode(str, Enum):
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_RATE_LIMIT = "PROVIDER_RATE_LIMIT"
    PROVIDER_AUTH_FAILED = "PROVIDER_AUTH_FAILED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    GUARDRAIL_BLOCKED = "GUARDRAIL_BLOCKED"
    TOOL_FETCH_FAILED = "TOOL_FETCH_FAILED"
    SEARCH_EXHAUSTED = "SEARCH_EXHAUSTED"
    GROUNDEDNESS_BELOW_THRESHOLD = "GROUNDEDNESS_BELOW_THRESHOLD"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    INTERNAL = "INTERNAL"


ERROR_SENTENCES: Dict[ErrorCode, Tuple[str, Severity]] = {
    ErrorCode.PROVIDER_TIMEOUT: ("The AI provider took too long to respond.", Severity.WARNING),
    ErrorCode.PROVIDER_RATE_LIMIT: ("The AI provider's rate limit was hit.", Severity.WARNING),
    ErrorCode.PROVIDER_AUTH_FAILED: ("The AI provider rejected our credentials.", Severity.ERROR),
    ErrorCode.PROVIDER_UNAVAILABLE: ("Every configured provider for this stage failed.", Severity.ERROR),
    ErrorCode.SCHEMA_VALIDATION_FAILED: ("The model's response didn't match the expected format.", Severity.WARNING),
    ErrorCode.GUARDRAIL_BLOCKED: ("A safety guardrail blocked this request.", Severity.ERROR),
    ErrorCode.TOOL_FETCH_FAILED: ("A research tool call failed.", Severity.WARNING),
    ErrorCode.SEARCH_EXHAUSTED: ("No further search results were available.", Severity.WARNING),
    ErrorCode.GROUNDEDNESS_BELOW_THRESHOLD: ("The report didn't meet the groundedness threshold.", Severity.WARNING),
    ErrorCode.BUDGET_EXCEEDED: ("The run exceeded its cost/token budget.", Severity.ERROR),
    ErrorCode.INTERNAL: ("An internal error occurred.", Severity.ERROR),
}


def sentence_for(code: ErrorCode) -> str:
    return ERROR_SENTENCES[code][0]


def severity_for(code: ErrorCode) -> Severity:
    return ERROR_SENTENCES[code][1]
```

```python
# observability/events.py
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from observability.node_state import NodeState


class RunEvent(BaseModel):
    run_id: str
    seq: int
    timestamp: datetime
    stage: str
    event_type: str
    node_state: Optional[NodeState] = None
    model_alias: Optional[str] = None
    resolved_deployment: Optional[str] = None
    attempt: Optional[int] = None
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    cache_hit: Optional[bool] = None
    message: str = ""
    detail: Optional[str] = None
```

```python
# observability/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_observability_types.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add observability/__init__.py observability/node_state.py observability/errors.py observability/events.py tests/test_observability_types.py
git commit -m "feat: add RunEvent, ErrorCode taxonomy, and NodeState types"
```

---

### Task 2: `EventBus` — stdout, per-run queue, and no-op trace sinks

**Files:**
- Create: `observability/bus.py`
- Test: `tests/test_bus.py`

**Interfaces:**
- Consumes: `observability.events.RunEvent` (Task 1).
- Produces: `observability.bus.EventBus` with `EventBus.instance() -> EventBus`
  (singleton), `EventBus.reset_for_tests() -> None`, `.emit(event: RunEvent) -> None`,
  `.queue_sink -> QueueSink`. `observability.bus.QueueSink` with `.get_queue(run_id:
  str) -> queue.Queue`, `.drop(run_id: str) -> None`. `observability.bus.StdoutSink`,
  `observability.bus.NoOpTraceSink` (both `.emit(event) -> None`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bus.py
from datetime import datetime

from observability.bus import EventBus, NoOpTraceSink, QueueSink, StdoutSink
from observability.events import RunEvent


def _event(run_id="r1", seq=1, event_type="stage_started") -> RunEvent:
    return RunEvent(run_id=run_id, seq=seq, timestamp=datetime.utcnow(),
                     stage="planner", event_type=event_type)


def test_instance_returns_singleton_with_default_sinks():
    EventBus.reset_for_tests()
    bus = EventBus.instance()
    assert bus is EventBus.instance()
    assert any(isinstance(s, StdoutSink) for s in bus.sinks)
    assert any(isinstance(s, QueueSink) for s in bus.sinks)
    assert any(isinstance(s, NoOpTraceSink) for s in bus.sinks)


def test_emit_delivers_to_queue_sink_isolated_by_run_id():
    EventBus.reset_for_tests()
    bus = EventBus.instance()
    bus.emit(_event(run_id="run-a", seq=1))
    bus.emit(_event(run_id="run-b", seq=1))

    got_a = bus.queue_sink.get_queue("run-a").get_nowait()
    assert got_a.run_id == "run-a"
    assert bus.queue_sink.get_queue("run-a").empty()
    assert bus.queue_sink.get_queue("run-b").qsize() == 1


def test_emit_survives_a_raising_sink_and_still_reaches_the_others():
    EventBus.reset_for_tests()
    bus = EventBus.instance()

    class BrokenSink:
        def emit(self, event):
            raise RuntimeError("boom")

    bus.sinks.insert(0, BrokenSink())
    bus.emit(_event(run_id="run-c"))

    assert bus.queue_sink.get_queue("run-c").get_nowait().run_id == "run-c"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'observability.bus'`

- [ ] **Step 3: Write the implementation**

```python
# observability/bus.py
"""
EventBus fans a RunEvent out to independent sinks. A sink failure is logged
and never propagates -- observability must never be able to break a
research run. See docs/superpowers/specs/2026-09-11-observability-event-bus-design.md.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Dict, List, Optional, Protocol

from observability.events import RunEvent

logger = logging.getLogger(__name__)


class Sink(Protocol):
    def emit(self, event: RunEvent) -> None: ...


class StdoutSink:
    """Logs one JSON line per event so any log drain (and `jq`) can read it."""

    def emit(self, event: RunEvent) -> None:
        logger.info(event.model_dump_json(exclude_none=True))


class QueueSink:
    """
    Per-run in-memory queue. A future Phase 3 SSE endpoint drains these;
    this phase only guarantees every event lands here. Unbounded: a run
    emits on the order of tens of events, not enough to matter yet.
    """

    def __init__(self) -> None:
        self._queues: Dict[str, "queue.Queue[RunEvent]"] = {}
        self._lock = threading.Lock()

    def get_queue(self, run_id: str) -> "queue.Queue[RunEvent]":
        with self._lock:
            return self._queues.setdefault(run_id, queue.Queue())

    def emit(self, event: RunEvent) -> None:
        self.get_queue(event.run_id).put(event)

    def drop(self, run_id: str) -> None:
        with self._lock:
            self._queues.pop(run_id, None)


class NoOpTraceSink:
    """Placeholder for Phase 7's Langfuse sink. Same interface, no-op body."""

    def emit(self, event: RunEvent) -> None:
        pass


class EventBus:
    _instance: Optional["EventBus"] = None

    def __init__(self) -> None:
        self.sinks: List[Sink] = [StdoutSink(), QueueSink(), NoOpTraceSink()]

    @classmethod
    def instance(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None

    def emit(self, event: RunEvent) -> None:
        for sink in self.sinks:
            try:
                sink.emit(event)
            except Exception:
                logger.exception("Sink %s failed to emit event", type(sink).__name__)

    @property
    def queue_sink(self) -> QueueSink:
        return next(s for s in self.sinks if isinstance(s, QueueSink))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bus.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add observability/bus.py tests/test_bus.py
git commit -m "feat: add EventBus with stdout, queue, and no-op trace sinks"
```

---

### Task 3: Thread-local `RunContext` correlation + `PipelineStageError.error_code`

**Files:**
- Create: `observability/context.py`
- Modify: `crew/research_crew.py:29-35` (the `PipelineStageError` class)
- Test: `tests/test_context.py`
- Test: `tests/test_pipeline_stage_error.py`

**Interfaces:**
- Consumes: `observability.bus.EventBus` (Task 2), `observability.events.RunEvent`
  (Task 1), `observability.errors.ErrorCode` (Task 1).
- Produces: `observability.context.start_run(run_id: str) -> None`,
  `.clear_run() -> None`, `.current_run() -> RunContext | None`,
  `.set_stage(stage: str) -> None`, `.emit_event(event_type: str, **fields) -> None`.
  `observability.context.RunContext` dataclass with fields `run_id: str, stage: str =
  "", seq: int = 0, stage_events: list[RunEvent] = []`. `crew.research_crew.
  PipelineStageError(stage: str, detail: str, error_code: ErrorCode = ErrorCode.
  INTERNAL)` — now exposes `.error_code`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_context.py
import threading
import time

from observability.bus import EventBus
from observability.context import clear_run, current_run, emit_event, set_stage, start_run


def test_emit_event_is_a_noop_with_no_active_run():
    clear_run()
    emit_event("stage_started")  # must not raise
    assert current_run() is None


def test_start_run_sets_context_and_emit_event_stamps_it():
    EventBus.reset_for_tests()
    start_run("run-1")
    set_stage("planner")
    emit_event("stage_started", message="go")

    ctx = current_run()
    assert ctx.run_id == "run-1"
    assert len(ctx.stage_events) == 1
    assert ctx.stage_events[0].event_type == "stage_started"
    assert ctx.stage_events[0].seq == 1
    clear_run()


def test_set_stage_resets_stage_events_for_the_new_stage():
    start_run("run-2")
    set_stage("planner")
    emit_event("stage_started")
    set_stage("researcher")
    assert current_run().stage_events == []
    clear_run()


def test_seq_increments_across_multiple_emits_within_a_run():
    start_run("run-3")
    set_stage("planner")
    emit_event("a")
    emit_event("b")
    assert [e.seq for e in current_run().stage_events] == [1, 2]
    clear_run()


def test_run_context_is_isolated_per_thread():
    results = {}

    def worker(run_id):
        start_run(run_id)
        set_stage("planner")
        emit_event("x")
        time.sleep(0.05)
        results[run_id] = current_run().run_id
        clear_run()

    t1 = threading.Thread(target=worker, args=("thread-a",))
    t2 = threading.Thread(target=worker, args=("thread-b",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert results == {"thread-a": "thread-a", "thread-b": "thread-b"}
```

```python
# tests/test_pipeline_stage_error.py
from crew.research_crew import PipelineStageError
from observability.errors import ErrorCode


def test_pipeline_stage_error_defaults_to_internal_error_code():
    err = PipelineStageError("planner", "boom")
    assert err.error_code == ErrorCode.INTERNAL
    assert err.stage == "planner"
    assert err.detail == "boom"


def test_pipeline_stage_error_accepts_explicit_error_code():
    err = PipelineStageError("researcher", "rate limited", ErrorCode.PROVIDER_RATE_LIMIT)
    assert err.error_code == ErrorCode.PROVIDER_RATE_LIMIT
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_context.py tests/test_pipeline_stage_error.py -v`
Expected: FAIL — `test_context.py` with `ModuleNotFoundError: No module named
'observability.context'`; `test_pipeline_stage_error.py` with `TypeError:
PipelineStageError.__init__() takes 3 positional arguments but 4 were given` (or
`AttributeError: 'PipelineStageError' object has no attribute 'error_code'`).

- [ ] **Step 3: Write the implementation**

```python
# observability/context.py
"""
Thread-local run correlation. Each pipeline run executes synchronously
start-to-finish on one worker thread (see docs/superpowers/specs/
2026-09-11-observability-event-bus-design.md), so a thread-local RunContext
gives every function on that thread ambient access to "which run is this"
without threading a parameter through get_llm()/RouterLLM/every agent
module.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from observability.bus import EventBus
from observability.events import RunEvent

logger = logging.getLogger(__name__)

_local = threading.local()


@dataclass
class RunContext:
    run_id: str
    stage: str = ""
    seq: int = 0
    stage_events: List[RunEvent] = field(default_factory=list)


def start_run(run_id: str) -> None:
    _local.ctx = RunContext(run_id=run_id)


def clear_run() -> None:
    _local.ctx = None


def current_run() -> Optional[RunContext]:
    return getattr(_local, "ctx", None)


def set_stage(stage: str) -> None:
    ctx = current_run()
    if ctx is not None:
        ctx.stage = stage
        ctx.stage_events = []


def emit_event(event_type: str, **fields) -> None:
    """
    Builds a RunEvent stamped with the active run's run_id/stage/seq and
    sends it to the EventBus. A no-op (besides a debug log line) if no run
    is active on this thread -- e.g. a direct/test call to a stage method
    outside ResearchCrew.run_with_meta.
    """
    ctx = current_run()
    if ctx is None:
        logger.debug("emit_event(%s) called with no active run context, dropped", event_type)
        return
    ctx.seq += 1
    event = RunEvent(
        run_id=ctx.run_id,
        seq=ctx.seq,
        timestamp=datetime.utcnow(),
        stage=ctx.stage,
        event_type=event_type,
        **fields,
    )
    ctx.stage_events.append(event)
    EventBus.instance().emit(event)
```

Modify `crew/research_crew.py` — add the import near the top (after the existing
`from pydantic import BaseModel` line) and replace the `PipelineStageError` class:

```python
from observability.errors import ErrorCode
```

```python
class PipelineStageError(RuntimeError):
    """Raised when a pipeline stage (an LLM call or its parsing) fails outright."""

    def __init__(self, stage: str, detail: str, error_code: ErrorCode = ErrorCode.INTERNAL):
        self.stage = stage
        self.detail = detail
        self.error_code = error_code
        super().__init__(f"Pipeline stage '{stage}' failed: {detail}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_context.py tests/test_pipeline_stage_error.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add observability/context.py crew/research_crew.py tests/test_context.py tests/test_pipeline_stage_error.py
git commit -m "feat: add thread-local RunContext correlation and PipelineStageError.error_code"
```

---

### Task 4: Attempt-history events from `RouterLLM` + pricing lookup

**Files:**
- Create: `observability/pricing.py`
- Modify: `llm/router.py` (add `get_deployment_aliases`)
- Modify: `llm/provider.py:146-161` (`get_llm` wires aliases into `RouterLLM`)
- Modify: `llm/router_llm.py` (aliases field, `classify_provider_exception`, instrument `call()`)
- Test: `tests/test_pricing.py`
- Test: extend `tests/test_router.py`
- Test: extend `tests/test_router_llm.py`

**Interfaces:**
- Consumes: `observability.context.emit_event` (Task 3), `observability.errors.
  ErrorCode` (Task 1).
- Produces: `observability.pricing.estimate_cost(provider_alias: str, model: str,
  prompt_tokens: int | None, completion_tokens: int | None) -> float | None`.
  `llm.router.get_deployment_aliases(agent_name: str) -> list[str]` (parallel to
  `get_deployments`). `llm.router_llm.classify_provider_exception(e: Exception) ->
  ErrorCode`. `RouterLLM` gains a pydantic field `aliases: list[str] = []`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pricing.py
from observability.pricing import estimate_cost


def test_free_alias_is_always_zero_regardless_of_model():
    assert estimate_cost("ollama_cloud", "gpt-oss:120b-cloud", 1000, 500) == 0.0


def test_free_suffix_model_is_zero_even_on_a_paid_alias():
    assert estimate_cost("openrouter", "nvidia/nemotron-3-super-120b-a12b:free", 1000, 500) == 0.0


def test_unknown_paid_model_returns_none_not_a_guess():
    assert estimate_cost("groq", "openai/gpt-oss-120b", 1000, 500) is None


def test_missing_token_counts_returns_none():
    assert estimate_cost("ollama_cloud", "gpt-oss:120b-cloud", None, None) is None
```

Add to `tests/test_router.py` (new function, existing tests untouched):

```python
def test_get_deployment_aliases_returns_config_aliases_parallel_to_deployments(monkeypatch):
    monkeypatch.setattr("llm.provider.settings.GROQ_API_KEY", "groq-test")
    monkeypatch.setattr("llm.provider.settings.OLLAMA_API_KEY", "ollama-test")
    monkeypatch.setattr("llm.provider.settings.OPENROUTER_API_KEY", "or-test")
    monkeypatch.setattr("llm.provider.settings.GEMINI_API_KEY", "gemini-test")

    import llm.router as router_module
    router_module._deployment_cache.clear()
    router_module._alias_cache.clear()

    deployments = router_module.get_deployments("researcher")
    aliases = router_module.get_deployment_aliases("researcher")
    assert len(aliases) == len(deployments)
    # Confirms aliases distinguish what deployments[i].provider cannot: Groq and
    # OpenRouter both build distinct crewai provider= strings here, but
    # ollama_local/ollama_cloud would collide on provider="ollama" if both were in
    # one chain -- aliases is the only place that distinction survives.
    assert aliases[0] == "openrouter"
    assert aliases[1] == "groq"
```

Add to `tests/test_router_llm.py` (new imports + new functions, existing tests and
`FakeDeployment`/`_router`/`_fake_response` untouched):

```python
from observability.bus import EventBus
from observability.context import clear_run, set_stage, start_run


def _drain(run_id):
    q = EventBus.instance().queue_sink.get_queue(run_id)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def test_successful_call_emits_llm_attempt_event_with_resolved_deployment():
    EventBus.reset_for_tests()
    start_run("run-events-1")
    set_stage("researcher")
    d1 = FakeDeployment("gpt-oss:120b-cloud", ["final answer"])
    router = RouterLLM(model="researcher", deployments=[d1], aliases=["ollama_cloud"])
    router.call("hi")
    clear_run()

    events = [e for e in _drain("run-events-1") if e.event_type == "llm_attempt"]
    assert len(events) == 1
    assert events[0].resolved_deployment == "ollama_cloud:gpt-oss:120b-cloud"
    assert events[0].attempt == 0
    assert events[0].message == "ok"


def test_fallback_emits_one_llm_attempt_event_per_attempt():
    EventBus.reset_for_tests()
    start_run("run-events-2")
    set_stage("researcher")
    rl_err = openai.RateLimitError("rate limited", response=_fake_response(429), body=None)
    d1 = FakeDeployment("a", [rl_err])
    d2 = FakeDeployment("b", ["from b"])
    router = RouterLLM(model="researcher", deployments=[d1, d2], aliases=["groq", "openrouter"])
    router.call("hi")
    clear_run()

    events = [e for e in _drain("run-events-2") if e.event_type == "llm_attempt"]
    assert len(events) == 2
    assert events[0].resolved_deployment == "groq:a"
    assert events[0].message == "PROVIDER_RATE_LIMIT"
    assert events[1].resolved_deployment == "openrouter:b"
    assert events[1].message == "ok"


def test_missing_aliases_falls_back_to_index_without_crashing():
    EventBus.reset_for_tests()
    start_run("run-events-3")
    set_stage("planner")
    d1 = FakeDeployment("a", ["final answer"])
    router = RouterLLM(model="planner", deployments=[d1])  # no aliases passed
    router.call("hi")
    clear_run()

    events = [e for e in _drain("run-events-3") if e.event_type == "llm_attempt"]
    assert events[0].resolved_deployment == "0:a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pricing.py tests/test_router.py tests/test_router_llm.py -v`
Expected: FAIL — `test_pricing.py` with `ModuleNotFoundError`; the new `test_router.py`
test with `AttributeError: module 'llm.router' has no attribute
'get_deployment_aliases'`; the new `test_router_llm.py` tests with `TypeError:
RouterLLM() got an unexpected keyword argument 'aliases'` (or `resolved_deployment`
mismatches once that's accepted).

- [ ] **Step 3: Verify `get_token_usage_summary()`'s return shape against the installed crewai**

Run:
```bash
python -c "
from llm.provider import get_llm
llm = get_llm('classifier').deployments[0]
llm.call('Reply with the single word: pong')
print(llm.get_token_usage_summary())
"
```
Expected: prints an object exposing `prompt_tokens`/`completion_tokens` (either
attributes on a `UsageMetrics`-shaped object, or dict keys). If the printed object
uses different names, use those names in `_usage_delta` in Step 4 below instead of
`prompt_tokens`/`completion_tokens`. This requires a configured API key for whichever
provider `classifier`'s first deployment resolves to; if none is set locally, skip the
live call and keep the dict-and-attribute-both-handled code in Step 4 as written — it
degrades to `None` token counts safely either way (see `test_missing_token_counts_
returns_none` above), which is the documented "never fabricate" behavior, not a bug.

- [ ] **Step 4: Write the implementation**

```python
# observability/pricing.py
"""
Static per-token pricing lookup for cost_usd on RunEvent("llm_attempt", ...).
Returns None rather than a fabricated number for any (provider, model) pair
whose real price hasn't been verified against the provider's own pricing
page yet.
"""
from typing import Dict, Optional, Tuple

# (config alias, model) -> (prompt $ / 1K tokens, completion $ / 1K tokens).
# groq/gemini models currently in config/agents_config.yaml are not priced
# here yet -- verify against each provider's live pricing page before adding
# an entry. Until then estimate_cost() correctly returns None for them
# rather than a guess.
_PRICING: Dict[Tuple[str, str], Tuple[float, float]] = {}

_FREE_ALIASES = frozenset({"ollama_local", "ollama_cloud"})


def estimate_cost(
    provider_alias: str,
    model: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Optional[float]:
    if prompt_tokens is None or completion_tokens is None:
        return None
    if provider_alias in _FREE_ALIASES or model.endswith(":free"):
        return 0.0
    rates = _PRICING.get((provider_alias, model))
    if rates is None:
        return None
    prompt_rate, completion_rate = rates
    return (prompt_tokens / 1000) * prompt_rate + (completion_tokens / 1000) * completion_rate
```

In `llm/router.py`, add near the other module-level caches (after `_deployment_cache`
/`_deployment_cache_lock`):

```python
_alias_cache: Dict[str, List[str]] = {}
```

Add this function after `get_deployments`:

```python
def get_deployment_aliases(agent_name: str) -> List[str]:
    """
    Ordered list of config-alias strings (e.g. "groq", "ollama_cloud") parallel to
    get_deployments(agent_name)'s LLM list -- needed because crewai.LLM's own
    `.provider` attribute collapses distinct aliases onto the same value (Groq and
    raw OpenAI both build provider="openai"; ollama_local and ollama_cloud both
    build provider="ollama"), so it can't identify which deployment actually
    served a call. Shares _deployment_cache_lock rather than adding a second lock.
    """
    with _deployment_cache_lock:
        if agent_name in _alias_cache:
            return _alias_cache[agent_name]
        config = _load_config()
        group = config.get("model_groups", {}).get(agent_name)
        if not group:
            raise LLMConfigError(
                f"no model_groups entry for agent_name={agent_name!r} in "
                "config/agents_config.yaml."
            )
        aliases = [entry["provider"] for entry in group]
        _alias_cache[agent_name] = aliases
        return aliases
```

In `llm/provider.py`, replace `get_llm`'s body:

```python
def get_llm(agent_name: Optional[str] = None) -> LLM:
    """
    Build the LLM for a given agent name (a top-level key in
    config/agents_config.yaml's model_groups section, e.g. "writer",
    "classifier"). Returns a RouterLLM wrapping that alias's ordered
    deployment list, so calls get conditional fallback across providers.
    """
    if not agent_name:
        raise LLMConfigError("get_llm() requires a non-empty agent_name.")

    from llm.router import get_deployment_aliases, get_deployments
    from llm.router_llm import RouterLLM

    deployments = get_deployments(agent_name)
    aliases = get_deployment_aliases(agent_name)
    return RouterLLM(model=agent_name, deployments=deployments, aliases=aliases)
```

In `llm/router_llm.py`: add imports, the `aliases` field, `classify_provider_exception`,
usage-delta helpers, and instrument `call()`. Add near the top imports:

```python
from typing import Any, List, Optional

from observability.context import emit_event
from observability.errors import ErrorCode
from observability.pricing import estimate_cost
```

Add after the existing exception tuples/constants (near `_GENERIC_STATUS_ERROR_ADVANCE_CODES`):

```python
def classify_provider_exception(e: Exception) -> ErrorCode:
    if isinstance(e, openai.AuthenticationError):
        return ErrorCode.PROVIDER_AUTH_FAILED
    if isinstance(e, openai.RateLimitError):
        return ErrorCode.PROVIDER_RATE_LIMIT
    if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
        return ErrorCode.PROVIDER_TIMEOUT
    if isinstance(e, (openai.PermissionDeniedError, openai.NotFoundError, openai.InternalServerError)):
        return ErrorCode.PROVIDER_UNAVAILABLE
    if isinstance(e, openai.APIStatusError) and e.status_code in _GENERIC_STATUS_ERROR_ADVANCE_CODES:
        return ErrorCode.PROVIDER_UNAVAILABLE
    return ErrorCode.INTERNAL


def _safe_usage(deployment) -> dict:
    try:
        summary = deployment.get_token_usage_summary()
    except Exception:
        return {}
    if hasattr(summary, "prompt_tokens"):
        return {"prompt_tokens": summary.prompt_tokens, "completion_tokens": summary.completion_tokens}
    if isinstance(summary, dict):
        return {"prompt_tokens": summary.get("prompt_tokens"), "completion_tokens": summary.get("completion_tokens")}
    return {}


def _usage_delta(before: dict, after: dict):
    def _diff(key):
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            return None
        return max(a - b, 0)
    return _diff("prompt_tokens"), _diff("completion_tokens")
```

Add the `aliases` field to the `RouterLLM` class, alongside `deployments`:

```python
class RouterLLM(BaseLLM):
    deployments: List[Any]
    aliases: List[str] = []

    _cooldown_until: dict = PrivateAttr(default_factory=dict)
```

Replace the entire body of `RouterLLM.call()`:

```python
    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        now = time.monotonic()
        order = [
            i for i in range(len(self.deployments))
            if self._cooldown_until.get(i, 0.0) <= now
        ]
        if not order:
            order = list(range(len(self.deployments)))

        last_error: Optional[Exception] = None
        for i in order:
            deployment = self.deployments[i]
            alias = self.aliases[i] if i < len(self.aliases) else str(i)
            resolved = f"{alias}:{getattr(deployment, 'model', i)}"
            for attempt in range(2):
                started = time.monotonic()
                usage_before = _safe_usage(deployment)
                try:
                    result = deployment.call(
                        messages, tools, callbacks, available_functions,
                        from_task, from_agent, response_model,
                    )
                except _TRANSIENT_EXCEPTIONS as e:
                    last_error = e
                    emit_event(
                        "llm_attempt", model_alias=self.model, resolved_deployment=resolved,
                        attempt=attempt, latency_ms=(time.monotonic() - started) * 1000,
                        message=classify_provider_exception(e).value, detail=str(e),
                    )
                    if attempt == 0:
                        logger.warning(
                            "Deployment %s transient error (attempt 1/2), retrying: %s",
                            getattr(deployment, "model", i), e,
                        )
                        time.sleep(_TRANSIENT_RETRY_BACKOFF_SECONDS)
                        continue
                    logger.warning(
                        "Deployment %s still failing after retry, advancing: %s",
                        getattr(deployment, "model", i), e,
                    )
                    self._cooldown_until[i] = time.monotonic() + _COOLDOWN_SECONDS
                    break
                except _IMMEDIATE_ADVANCE_EXCEPTIONS as e:
                    last_error = e
                    emit_event(
                        "llm_attempt", model_alias=self.model, resolved_deployment=resolved,
                        attempt=attempt, latency_ms=(time.monotonic() - started) * 1000,
                        message=classify_provider_exception(e).value, detail=str(e),
                    )
                    logger.warning(
                        "Deployment %s unavailable, advancing immediately: %s",
                        getattr(deployment, "model", i), e,
                    )
                    self._cooldown_until[i] = time.monotonic() + _COOLDOWN_SECONDS
                    break
                except openai.APIStatusError as e:
                    if e.status_code not in _GENERIC_STATUS_ERROR_ADVANCE_CODES:
                        raise
                    last_error = e
                    emit_event(
                        "llm_attempt", model_alias=self.model, resolved_deployment=resolved,
                        attempt=attempt, latency_ms=(time.monotonic() - started) * 1000,
                        message=classify_provider_exception(e).value, detail=str(e),
                    )
                    logger.warning(
                        "Deployment %s unavailable (status %s), advancing immediately: %s",
                        getattr(deployment, "model", i), e.status_code, e,
                    )
                    self._cooldown_until[i] = time.monotonic() + _COOLDOWN_SECONDS
                    break
                else:
                    if isinstance(result, str):
                        _raise_if_refusal(result)
                    usage_after = _safe_usage(deployment)
                    prompt_tok, completion_tok = _usage_delta(usage_before, usage_after)
                    emit_event(
                        "llm_attempt", model_alias=self.model, resolved_deployment=resolved,
                        attempt=attempt, latency_ms=(time.monotonic() - started) * 1000,
                        prompt_tokens=prompt_tok, completion_tokens=completion_tok,
                        cost_usd=estimate_cost(alias, getattr(deployment, "model", ""), prompt_tok, completion_tok),
                        message="ok",
                    )
                    return result

        if last_error is not None:
            raise last_error
        raise RuntimeError("RouterLLM: no deployments configured")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_pricing.py tests/test_router.py tests/test_router_llm.py -v`
Expected: all passed (4 new + 3 new + all pre-existing `test_router_llm.py`/
`test_router.py` cases still green)

- [ ] **Step 6: Run the full suite to confirm no regression**

Run: `pytest -q`
Expected: all passed

- [ ] **Step 7: Commit**

```bash
git add observability/pricing.py llm/router.py llm/provider.py llm/router_llm.py tests/test_pricing.py tests/test_router.py tests/test_router_llm.py
git commit -m "feat: emit llm_attempt events with resolved deployment, latency, and cost"
```

---

### Task 5: `PipelineRun`/`PipelineStageState` DB tables + repository functions

**Files:**
- Modify: `db/models.py` (append two new model classes)
- Modify: `db/repository.py` (append new functions)
- Test: `tests/test_pipeline_run_repository.py`

**Interfaces:**
- Produces: `db.models.PipelineRun`, `db.models.PipelineStageState`.
  `db.repository.create_pipeline_run(run_id: str, question_id: int | None, topic: str)
  -> None`, `.reattach_pipeline_run(run_id: str, question_id: int) -> None`,
  `.set_run_status(run_id: str, status: str) -> None`, `.set_stage_state(run_id: str,
  stage: str, state: str, detail: str | None = None, attempts: list[dict] | None =
  None) -> None`, `.get_run(run_id: str) -> dict | None`, `.get_stage_states(run_id:
  str) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_run_repository.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_run_repository.py -v`
Expected: FAIL with `AttributeError: module 'db.repository' has no attribute
'create_pipeline_run'`

- [ ] **Step 3: Write the implementation**

Append to `db/models.py` (after the `Conclusion` class):

```python
class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    run_id = Column(String, primary_key=True)
    question_id = Column(Integer, ForeignKey("research_questions.id"), nullable=True)
    topic = Column(String, nullable=False)
    status = Column(String, nullable=False, default="running")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class PipelineStageState(Base):
    __tablename__ = "pipeline_stage_states"

    id = Column(Integer, primary_key=True)
    run_id = Column(String, ForeignKey("pipeline_runs.run_id"), nullable=False)
    stage = Column(String, nullable=False)
    state = Column(String, nullable=False)
    attempts = Column(JSON, default=list, nullable=False)
    detail = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
```

Append to `db/repository.py` (update the top import line to add the two new models,
and add `from datetime import datetime`):

```python
from datetime import datetime

from db.models import (
    Conclusion, Contradiction, Finding, PipelineRun, PipelineStageState,
    ResearchQuestion, Source,
)
```

```python
_TERMINAL_STAGE_STATES = {"succeeded", "degraded", "failed", "not_reached", "skipped_cached"}


def create_pipeline_run(run_id: str, question_id: Optional[int], topic: str) -> None:
    with get_session() as session:
        session.add(PipelineRun(run_id=run_id, question_id=question_id, topic=topic, status="running"))


def reattach_pipeline_run(run_id: str, question_id: int) -> None:
    """
    Ensures a pipeline_runs row exists for run_id, linked to question_id --
    used when resuming a run under the same run_id after a prior failed attempt.
    """
    with get_session() as session:
        run = session.get(PipelineRun, run_id)
        if run is None:
            session.add(PipelineRun(run_id=run_id, question_id=question_id, topic="", status="running"))
        else:
            run.question_id = question_id
            run.status = "running"
            run.updated_at = datetime.utcnow()


def set_run_status(run_id: str, status: str) -> None:
    with get_session() as session:
        run = session.get(PipelineRun, run_id)
        if run is not None:
            run.status = status
            run.updated_at = datetime.utcnow()


def set_stage_state(
    run_id: str,
    stage: str,
    state: str,
    detail: Optional[str] = None,
    attempts: Optional[List[dict]] = None,
) -> None:
    """
    Get-or-create by (run_id, stage): a stage transitions pending -> running ->
    a terminal state on the same row, not a new row per transition.
    """
    with get_session() as session:
        row = (
            session.query(PipelineStageState)
            .filter(PipelineStageState.run_id == run_id, PipelineStageState.stage == stage)
            .first()
        )
        now = datetime.utcnow()
        if row is None:
            row = PipelineStageState(run_id=run_id, stage=stage, state=state, attempts=attempts or [])
            session.add(row)
        else:
            row.state = state
            if attempts is not None:
                row.attempts = attempts
        if detail is not None:
            row.detail = detail
        if state == "running":
            row.started_at = now
        elif state in _TERMINAL_STAGE_STATES:
            row.finished_at = now


def get_run(run_id: str) -> Optional[dict]:
    with get_session() as session:
        run = session.get(PipelineRun, run_id)
        if run is None:
            return None
        return {
            "run_id": run.run_id,
            "question_id": run.question_id,
            "topic": run.topic,
            "status": run.status,
            "created_at": run.created_at.isoformat(),
            "updated_at": run.updated_at.isoformat(),
        }


def get_stage_states(run_id: str) -> List[dict]:
    with get_session() as session:
        rows = (
            session.query(PipelineStageState)
            .filter(PipelineStageState.run_id == run_id)
            .order_by(PipelineStageState.id)
            .all()
        )
        return [
            {
                "stage": r.stage,
                "state": r.state,
                "attempts": r.attempts or [],
                "detail": r.detail,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            }
            for r in rows
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_run_repository.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -q`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add db/models.py db/repository.py tests/test_pipeline_run_repository.py
git commit -m "feat: add pipeline_runs/pipeline_stage_states tables and repository functions"
```

---

### Task 6: Stage-dispatch loop, node states, and resume in `ResearchCrew`

**Files:**
- Modify: `crew/research_crew.py` (imports, `_extract_pydantic`, all five `_run_*_stage`
  methods' error-raising, `__init__`, replace `run_with_meta`, add dispatch wrappers)
- Test: `tests/test_research_crew_stage_dispatch.py`

**Interfaces:**
- Consumes: `observability.context.{start_run, clear_run, current_run, set_stage,
  emit_event}` (Task 3), `observability.errors.ErrorCode` (Task 1),
  `observability.node_state.NodeState` (Task 1), `llm.router_llm.
  classify_provider_exception` (Task 4), `db.repository.{create_pipeline_run,
  reattach_pipeline_run, set_run_status, set_stage_state, get_run, get_stage_states}`
  (Task 5).
- Produces: `ResearchCrew.run_with_meta(topic: str, force_rerun: bool = False, run_id:
  str | None = None, resume_question_id: int | None = None) -> dict` — the returned
  dict now also carries `"run_id"`. `ResearchCrew._run_research_stage(question_id,
  sub_questions) -> tuple[list[dict], bool]` (was `-> list[dict]`; second element is
  `degraded`). `ResearchCrew.STAGES: list[str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_research_crew_stage_dispatch.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_research_crew_stage_dispatch.py -v`
Expected: FAIL — `run_with_meta()` does not accept `run_id`/`resume_question_id`
keyword arguments yet (`TypeError`).

- [ ] **Step 3: Write the implementation**

In `crew/research_crew.py`, add to the imports (near the top, after the existing
`from tools.file_tool import save_report` line):

```python
import uuid

from llm.router_llm import classify_provider_exception
from observability.context import clear_run, current_run, emit_event, set_stage, start_run
from observability.node_state import NodeState
```

(`from observability.errors import ErrorCode` was already added in Task 3.)

Update both raise sites in `_extract_pydantic` to pass `ErrorCode.SCHEMA_VALIDATION_FAILED`:

```python
def _extract_pydantic(crew_output, model: Type[T], stage: str) -> T:
    if getattr(crew_output, "pydantic", None) is not None:
        return crew_output.pydantic

    raw = str(crew_output).strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return model.model_validate_json(match.group(0))
        except Exception as e:
            raise PipelineStageError(
                stage, f"could not parse structured output: {e}", ErrorCode.SCHEMA_VALIDATION_FAILED
            ) from e

    raise PipelineStageError(
        stage, "LLM did not return the expected structured output", ErrorCode.SCHEMA_VALIDATION_FAILED
    )
```

Replace `ResearchCrew.__init__`:

```python
class ResearchCrew:
    STAGES = ["planner", "researcher", "evidence_comparator", "classifier", "writer"]

    def __init__(self):
        init_db()
        self._reuse_checker = KnowledgeReuseChecker()
        self._STAGE_RUNNERS = {
            "planner": self._dispatch_planner,
            "researcher": self._dispatch_researcher,
            "evidence_comparator": self._dispatch_evidence_comparator,
            "classifier": self._dispatch_classifier,
            "writer": self._dispatch_writer,
        }
```

Replace `run_with_meta` (keep `run` and `_try_reuse` exactly as they are):

```python
    def run_with_meta(
        self,
        topic: str,
        force_rerun: bool = False,
        run_id: Optional[str] = None,
        resume_question_id: Optional[int] = None,
    ) -> dict:
        run_id = run_id or str(uuid.uuid4())
        start_run(run_id)
        try:
            emit_event("run_started", message=f"topic={topic!r}")

            if resume_question_id is None:
                if not force_rerun:
                    reused = self._try_reuse(topic)
                    if reused is not None:
                        reused["run_id"] = run_id
                        emit_event("run_finished", cache_hit=True, message="reused prior research")
                        return reused

                question_id = repository.create_question(topic)
                logger.info("Created research_question id=%s for topic=%r", question_id, topic)
                repository.create_pipeline_run(run_id, question_id, topic)
                stage_outputs: dict = {}
                start_index = 0
            else:
                question_id = resume_question_id
                repository.reattach_pipeline_run(run_id, question_id)
                stage_outputs = self._load_resumable_state(question_id)
                start_index = self._first_unresolved_stage_index(run_id)

            for i, stage in enumerate(self.STAGES):
                set_stage(stage)
                if i < start_index:
                    repository.set_stage_state(run_id, stage, NodeState.SKIPPED_CACHED.value)
                    continue

                repository.set_stage_state(run_id, stage, NodeState.RUNNING.value)
                emit_event("stage_started")
                try:
                    stage_outputs = self._STAGE_RUNNERS[stage](topic, question_id, stage_outputs)
                except PipelineStageError as e:
                    attempts = self._collect_attempt_history()
                    repository.set_stage_state(
                        run_id, stage, NodeState.FAILED.value, detail=e.detail, attempts=attempts
                    )
                    emit_event("stage_failed", detail=e.detail)
                    for remaining in self.STAGES[i + 1:]:
                        repository.set_stage_state(run_id, remaining, NodeState.NOT_REACHED.value)
                    repository.set_run_status(run_id, "failed")
                    raise

                state = NodeState.DEGRADED if stage_outputs.pop("_degraded", False) else NodeState.SUCCEEDED
                attempts = self._collect_attempt_history()
                repository.set_stage_state(run_id, stage, state.value, attempts=attempts)
                emit_event("stage_finished", node_state=state)

            findings = stage_outputs["findings"]
            contradictions = stage_outputs["contradictions"]
            report = stage_outputs["report"]

            conclusion_text = _extract_conclusion_section(report)
            conclusion_id = repository.add_conclusion(
                question_id, conclusion_text, [f["id"] for f in findings]
            )

            filepath = save_report(report, topic)
            logger.info("Report saved to: %s", filepath)

            self._index_for_reuse(question_id, topic, conclusion_text, findings, conclusion_id)
            repository.set_run_status(run_id, "succeeded")
            emit_event("run_finished")

            return {
                "report": report,
                "question_id": question_id,
                "run_id": run_id,
                "conclusion_id": conclusion_id,
                "reused": False,
                "saved_to": filepath,
                "contradictions_count": len(contradictions),
                "findings_count": len(findings),
            }
        finally:
            clear_run()

    def _collect_attempt_history(self) -> List[dict]:
        ctx = current_run()
        if ctx is None:
            return []
        return [e.model_dump(mode="json") for e in ctx.stage_events if e.event_type == "llm_attempt"]

    def _load_resumable_state(self, question_id: int) -> dict:
        question = repository.get_question(question_id)
        return {
            "sub_questions": question["sub_questions"] if question else [],
            "findings": repository.get_findings_list(question_id),
            "contradictions": repository.get_contradictions(question_id),
        }

    def _first_unresolved_stage_index(self, run_id: str) -> int:
        states = {s["stage"]: s["state"] for s in repository.get_stage_states(run_id)}
        resolved = {"succeeded", "degraded", "skipped_cached"}
        for i, stage in enumerate(self.STAGES):
            if states.get(stage) not in resolved:
                return i
        return len(self.STAGES)

    # ── Stage dispatch wrappers ─────────────────────────────────────
    # Each wrapper adapts one _run_*_stage method's own (differing) signature to
    # the uniform (topic, question_id, stage_outputs) -> stage_outputs shape the
    # dispatch loop above calls through self._STAGE_RUNNERS.

    def _dispatch_planner(self, topic, question_id, stage_outputs):
        sub_questions = self._run_planning_stage(topic)
        repository.set_sub_questions(question_id, sub_questions)
        stage_outputs["sub_questions"] = sub_questions
        return stage_outputs

    def _dispatch_researcher(self, topic, question_id, stage_outputs):
        findings, degraded = self._run_research_stage(question_id, stage_outputs["sub_questions"])
        stage_outputs["findings"] = findings
        stage_outputs["_degraded"] = degraded
        return stage_outputs

    def _dispatch_evidence_comparator(self, topic, question_id, stage_outputs):
        contradictions = self._run_comparison_stage(question_id, stage_outputs["findings"])
        stage_outputs["contradictions"] = contradictions
        return stage_outputs

    def _dispatch_classifier(self, topic, question_id, stage_outputs):
        findings = self._run_classification_stage(question_id, stage_outputs["findings"])
        stage_outputs["findings"] = findings
        return stage_outputs

    def _dispatch_writer(self, topic, question_id, stage_outputs):
        report = self._run_writing_stage(topic, stage_outputs["findings"], stage_outputs["contradictions"])
        stage_outputs["report"] = report
        return stage_outputs
```

Update `_run_planning_stage`'s exception branch to classify the error code:

```python
    def _run_planning_stage(self, topic: str) -> List[str]:
        planner = get_planner()
        plan_task = get_plan_task(topic, planner)
        try:
            result = _run_single_task_crew(planner, plan_task)
        except Exception as e:
            logger.error("Planner stage failed: %s", e)
            raise PipelineStageError("planner", str(e), classify_provider_exception(e)) from e

        sub_questions = _parse_sub_questions(str(result))
        if not sub_questions:
            raise PipelineStageError("planner", "no sub-questions were produced", ErrorCode.SCHEMA_VALIDATION_FAILED)
        return sub_questions
```

In `_run_research_stage`, update the provider-unavailable raise and the return
statement (everything else in the method is unchanged):

```python
                except Exception as e:
                    if _is_provider_unavailable_error(e):
                        logger.error("Researcher stage failed (provider unavailable): %s", e)
                        raise PipelineStageError("researcher", str(e), classify_provider_exception(e)) from e
```

```python
        if not any_succeeded:
            raise PipelineStageError(
                "researcher", "no sub-question produced a parseable finding",
                ErrorCode.SCHEMA_VALIDATION_FAILED,
            )

        findings = repository.get_findings_list(question_id)
        degraded = any(f["source_url"] == "none" for f in findings)
        return findings, degraded
```

Update `_run_comparison_stage`, `_run_classification_stage`, and `_run_writing_stage`'s
`except Exception as e:` blocks the same way, e.g.:

```python
    def _run_comparison_stage(self, question_id: int, findings: List[dict]) -> List[dict]:
        comparator = get_evidence_comparator()
        grouped = repository.get_findings_grouped(question_id)
        compare_task = get_compare_task(comparator, grouped)
        try:
            result = _run_single_task_crew(comparator, compare_task)
        except Exception as e:
            logger.error("EvidenceComparator stage failed: %s", e)
            raise PipelineStageError("evidence_comparator", str(e), classify_provider_exception(e)) from e
        ...
```

(same one-line change to the `raise PipelineStageError(...)` call in
`_run_classification_stage` and `_run_writing_stage` — pass `classify_provider_
exception(e)` as the third argument, nothing else in those methods changes)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_research_crew_stage_dispatch.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -q`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add crew/research_crew.py tests/test_research_crew_stage_dispatch.py
git commit -m "feat: stage-dispatch loop with node states and resume in ResearchCrew"
```

---

### Task 7: API wiring — `run_id` correlation, `GET /research/{run_id}`, resume endpoint

**Files:**
- Modify: `worker/job_manager.py` (`submit` gains an optional `job_id` parameter)
- Modify: `app.py` (`run_id` generation, `ResearchResponse.run_id`, two new endpoints)
- Test: `tests/test_job_manager.py`
- Test: `tests/test_run_endpoints.py`

**Interfaces:**
- Consumes: `db.repository.{get_run, get_stage_states}` (Task 5),
  `crew.research_crew.ResearchCrew.run_with_meta` (Task 6).
- Produces: `worker.job_manager.submit(fn, *args, job_id: str | None = None, **kwargs)
  -> str`. New routes `GET /research/{run_id}` and `POST /research/{run_id}/resume`.
  `ResearchResponse` gains `run_id: str | None = None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_job_manager.py
import time

from worker import job_manager


def test_submit_uses_given_job_id_when_provided():
    job_id = job_manager.submit(lambda: "result", job_id="custom-id")
    assert job_id == "custom-id"
    for _ in range(50):
        job = job_manager.get_job(job_id)
        if job["status"] == "done":
            break
        time.sleep(0.02)
    assert job["result"] == "result"


def test_submit_generates_a_job_id_when_not_provided():
    job_id = job_manager.submit(lambda: "result")
    assert job_id
    assert job_manager.get_job(job_id) is not None
```

```python
# tests/test_run_endpoints.py
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app as app_module
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_job_manager.py tests/test_run_endpoints.py -v`
Expected: FAIL — `test_job_manager.py`'s first test with `TypeError: submit() got an
unexpected keyword argument 'job_id'`; `test_run_endpoints.py`'s `GET`/`resume` tests
with 404s for routes that don't exist yet.

- [ ] **Step 3: Write the implementation**

In `worker/job_manager.py`, replace `submit`:

```python
def submit(fn: Callable[..., Any], *args, job_id: Optional[str] = None, **kwargs) -> str:
    """
    Submit fn(*args, **kwargs) to run in the background. Returns a job_id --
    the caller's own job_id when given (e.g. the pipeline's run_id, so
    GET /jobs/{job_id} and GET /research/{run_id} address the same run), or a
    freshly generated one otherwise.
    """
    job_id = job_id or str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {"status": "pending", "result": None, "error": None}

    def _run():
        with _lock:
            _jobs[job_id]["status"] = "running"
        try:
            result = fn(*args, **kwargs)
            with _lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = result
        except Exception as e:
            logger.error("Background job %s failed: %s", job_id, e)
            with _lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(e)

    _executor.submit(_run)
    return job_id
```

In `app.py`, add `import uuid` to the top imports. Update `_run_pipeline` to accept an
optional `run_id` (so `/research/stream`'s existing call site keeps working unchanged):

```python
def _run_pipeline(topic: str, force_rerun: bool, run_id: str | None = None) -> dict:
    crew = ResearchCrew()
    return crew.run_with_meta(topic, force_rerun=force_rerun, run_id=run_id)
```

Add `run_id` to `ResearchResponse`:

```python
class ResearchResponse(BaseModel):
    topic: str
    report: str
    saved_to: str | None = None
    question_id: int | None = None
    run_id: str | None = None
    reused: bool = False
    contradictions_count: int = 0
    findings_count: int = 0
```

Replace `run_research`'s body to generate and pass `run_id`, and include it in the
response:

```python
@app.post("/research", response_model=ResearchResponse)
async def run_research(req: ResearchRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    run_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _run_pipeline, req.topic, req.force_rerun, run_id)
    except PipelineStageError as e:
        raise HTTPException(status_code=502, detail=f"Pipeline stage '{e.stage}' failed: {e.detail}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return ResearchResponse(
        topic=req.topic,
        report=result["report"],
        saved_to=result.get("saved_to"),
        question_id=result.get("question_id"),
        run_id=result.get("run_id", run_id),
        reused=result.get("reused", False),
        contradictions_count=result.get("contradictions_count", 0),
        findings_count=result.get("findings_count", 0),
    )
```

Replace `run_research_async`'s body:

```python
@app.post("/research/async")
async def run_research_async(req: ResearchRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    run_id = str(uuid.uuid4())
    job_manager.submit(_run_pipeline, req.topic, req.force_rerun, run_id, job_id=run_id)
    return {"job_id": run_id, "status": "pending"}
```

Add these two new routes after `get_job`:

```python
@app.get("/research/{run_id}")
def get_run(run_id: str):
    """Node-state view of one run, for polling without SSE."""
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run_id.")
    return {**run, "stages": repository.get_stage_states(run_id)}


@app.post("/research/{run_id}/resume", response_model=ResearchResponse)
async def resume_research(run_id: str):
    """Re-enters the pipeline at the first non-succeeded stage of a prior run."""
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run_id.")
    if run["status"] == "succeeded":
        raise HTTPException(status_code=409, detail="Run already succeeded; nothing to resume.")

    crew = ResearchCrew()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, crew.run_with_meta, run["topic"], False, run_id, run["question_id"]
        )
    except PipelineStageError as e:
        raise HTTPException(status_code=502, detail=f"Pipeline stage '{e.stage}' failed: {e.detail}")

    return ResearchResponse(
        topic=run["topic"],
        report=result["report"],
        saved_to=result.get("saved_to"),
        question_id=result.get("question_id"),
        run_id=result.get("run_id", run_id),
        reused=result.get("reused", False),
        contradictions_count=result.get("contradictions_count", 0),
        findings_count=result.get("findings_count", 0),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_job_manager.py tests/test_run_endpoints.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -q`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add worker/job_manager.py app.py tests/test_job_manager.py tests/test_run_endpoints.py
git commit -m "feat: correlate run_id/job_id and add GET /research/{run_id} + resume endpoint"
```

---

## Final verification (after Task 7)

- [ ] Run the full test suite once more: `pytest -q` — expect all green, no skips.
- [ ] Manual smoke test: start the server (`uvicorn app:app --reload` or the project's
  existing run method), `POST /research` with a real topic, confirm the response
  includes a `run_id`, then `GET /research/{run_id}` and confirm `stages` shows all
  five stages `succeeded` with non-empty `attempts` arrays containing real
  `resolved_deployment`/`latency_ms` values from the actual run.
- [ ] Confirm stdout logs during that run contain one JSON line per `RunEvent`
  (`grep run_started` / `grep llm_attempt` on the server's stdout).
