# Phase 2 — Observability spine: event bus, error taxonomy, node states, resume

Sub-project of `research-pipeline-production-upgrade.md` (Phase 2, §5). Scope: give the
pipeline a `RunEvent` stream, a closed error taxonomy, an explicit per-stage state
machine, and a resume path — the substrate the UI (Phase 8) and real streaming (Phase 3)
both render on top of. This pass does **not** rewrite `/research/stream` into real SSE
(that is Phase 3) and does **not** wire Langfuse (Phase 7) — both get a stub sink that
Phase 2's plumbing already reaches, so later phases fill in behavior, not wiring.

## Constraints carried over from Phase 1 / the parent doc

- `get_llm(agent_name)` stays the only interface agent code calls. No agent module
  (`agents/*.py`, `tasks/*.py`) may change signature or gain a new parameter.
- `crew/research_crew.py`'s existing stage bodies (`_run_planning_stage`,
  `_run_research_stage`, etc.), the per-sub-question retry/placeholder logic, and
  `_extract_pydantic`'s raw-JSON parsing are correct and are not being rewritten — they
  get wrapped by a stage-dispatch loop, not replaced.
- `PipelineStageError(stage, detail)` is extended, not replaced.

## Architecture

### The correlation problem

Each pipeline run executes synchronously, start to finish, inside one worker thread
(`asyncio.loop.run_in_executor` for `/research`, `worker.job_manager`'s
`ThreadPoolExecutor` for `/research/async`). A thread is never shared between two
concurrent runs — the executor doesn't hand a second job to a thread until the first
finishes. That makes `threading.local` a correct, zero-dependency way to give every
function on that thread ambient access to "which run is this," without changing any
agent-facing signature.

Rejected alternative: threading a `RunContext`/`EventBus` parameter through
`get_llm()` → `RouterLLM` → every agent factory → every task module. Same information,
but breaks the Phase 1 contract that `get_llm(agent_name)` is the sole public interface,
and touches ~15 call sites for no behavioral gain over the thread-local approach.

### `observability/context.py`

```python
_local = threading.local()

@dataclass
class RunContext:
    run_id: str
    stage: str = ""
    _seq: int = 0

def start_run(run_id: str) -> None:
    _local.ctx = RunContext(run_id=run_id)

def clear_run() -> None:
    _local.ctx = None

def current_run() -> RunContext | None:
    return getattr(_local, "ctx", None)

def set_stage(stage: str) -> None:
    ctx = current_run()
    if ctx:
        ctx.stage = stage

def emit_event(event_type: str, **fields) -> None:
    ctx = current_run()
    if ctx is None:
        logger.debug("emit_event(%s) called with no active run context, dropped", event_type)
        return
    ctx._seq += 1
    event = RunEvent(run_id=ctx.run_id, seq=ctx._seq, stage=ctx.stage,
                      event_type=event_type, timestamp=datetime.utcnow(), **fields)
    EventBus.instance().emit(event)
```

`ResearchCrew.run_with_meta` calls `start_run(run_id)` at the top and `clear_run()` in a
`finally`. Every other call site (`RouterLLM.call()`, the stage-dispatch loop) just calls
`emit_event(...)` — no context object passed around.

### `observability/events.py` — `RunEvent`

Exact shape from the parent doc §5.1:

```python
class RunEvent(BaseModel):
    run_id: str
    seq: int
    timestamp: datetime
    stage: str
    event_type: str          # run_started | stage_started | stage_finished |
                              # stage_failed | llm_attempt | cache_hit |
                              # guardrail_fired | run_finished | run_aborted
    node_state: NodeState | None = None
    model_alias: str | None = None
    resolved_deployment: str | None = None
    attempt: int | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    cache_hit: bool | None = None
    message: str = ""
    detail: str | None = None    # server-only; sinks decide whether to forward it
```

`event_type` is a plain string, not an enum — the set is expected to grow across Phase
3/4/5 (guardrail_fired, cache_hit variants) and a string keeps those additions from
touching this file.

### `observability/errors.py` — `ErrorCode`

```python
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

ERROR_SENTENCES: dict[ErrorCode, tuple[str, Severity]] = {
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
```

`PipelineStageError` gains `error_code: ErrorCode = ErrorCode.INTERNAL` as a third
constructor argument (default keeps every existing `PipelineStageError(stage, detail)`
call site compiling unchanged; call sites are updated to pass the right code where it's
cheap and obvious — provider-unavailable branches get `PROVIDER_UNAVAILABLE`, parse
failures get `SCHEMA_VALIDATION_FAILED` — but this is not a mechanical blanket rename).

Mapping from caught exceptions to `ErrorCode` lives next to where they're already
classified: `llm/router_llm.py` already sorts exceptions into
`_TRANSIENT_EXCEPTIONS`/`_IMMEDIATE_ADVANCE_EXCEPTIONS`/status-code buckets — add a
small `classify_provider_exception(e) -> ErrorCode` there and reuse it from both
`RouterLLM.call()` (for the `llm_attempt` event's `detail`) and
`research_crew._is_provider_unavailable_error`'s call sites.

### `observability/node_state.py` — `NodeState`

```python
class NodeState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    DEGRADED = "degraded"
    FAILED = "failed"
    NOT_REACHED = "not_reached"
    SKIPPED_CACHED = "skipped_cached"
```

### `observability/bus.py` — `EventBus`

```python
class Sink(Protocol):
    def emit(self, event: RunEvent) -> None: ...

class StdoutSink:
    def emit(self, event):
        logger.info(event.model_dump_json(exclude_none=True))

class QueueSink:
    """Per-run in-memory queue. Phase 3's SSE endpoint drains these; this
    phase only guarantees every event lands here."""
    def __init__(self):
        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    def get_queue(self, run_id: str) -> queue.Queue:
        with self._lock:
            return self._queues.setdefault(run_id, queue.Queue())

    def emit(self, event):
        self.get_queue(event.run_id).put(event)

    def drop(self, run_id: str) -> None:
        with self._lock:
            self._queues.pop(run_id, None)

class NoOpTraceSink:
    """Placeholder for Phase 7's Langfuse sink — same interface, no-op body."""
    def emit(self, event): pass

class EventBus:
    _instance: "EventBus | None" = None

    def __init__(self):
        self.sinks: list[Sink] = [StdoutSink(), QueueSink(), NoOpTraceSink()]

    @classmethod
    def instance(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

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

A sink failure never raises into the pipeline — observability must not be able to break
a research run. `QueueSink` is unbounded for this phase (a run emits on the order of
tens of events; bounding it is a Phase 3/9 concern once real consumers exist).

### Attempt history — `RouterLLM.call()`

The existing per-attempt loop already has every value needed; it gets timed and now
emits instead of only logging:

```python
for attempt in range(2):
    started = time.monotonic()
    usage_before = _safe_usage(deployment)
    try:
        result = deployment.call(...)
    except _TRANSIENT_EXCEPTIONS as e:
        emit_event("llm_attempt", model_alias=self.model,
                    resolved_deployment=getattr(deployment, "model", str(i)),
                    attempt=attempt, latency_ms=(time.monotonic() - started) * 1000,
                    detail=f"{classify_provider_exception(e).value}: {e}")
        ...  # existing retry/advance logic unchanged
    else:
        usage_after = _safe_usage(deployment)
        prompt_tok, completion_tok = _usage_delta(usage_before, usage_after)
        emit_event("llm_attempt", model_alias=self.model,
                    resolved_deployment=getattr(deployment, "model", str(i)),
                    attempt=attempt, latency_ms=(time.monotonic() - started) * 1000,
                    prompt_tokens=prompt_tok, completion_tokens=completion_tok,
                    cost_usd=estimate_cost(deployment, prompt_tok, completion_tok),
                    message="ok")
        ...  # existing refusal check + return
```

`_safe_usage`/`_usage_delta` wrap `deployment.get_token_usage_summary()` — **the exact
shape of that return value needs empirical verification against the installed crewai
1.14.2 during implementation** (same kind of check Phase 1's task briefs already did for
`supports_function_calling`/`APIStatusError`); if it turns out not to expose a
per-call-diffable counter, fall back to emitting `prompt_tokens=None,
completion_tokens=None` rather than a wrong number — the doc's principle 1 only requires
that *when* a number is shown it be real, not that every event carries one.

### `observability/pricing.py`

A static `dict[tuple[provider, model], (prompt_$_per_1k, completion_$_per_1k)]` covering
the models actually configured in `config/agents_config.yaml`'s `model_groups`.
`estimate_cost(deployment, prompt_tok, completion_tok) -> float | None` returns `None`
for any (provider, model) not in the table — never a fabricated number. Free-tier /
local (`ollama_local`, `ollama_cloud`, OpenRouter `:free` models) always resolve to `0.0`
explicitly rather than falling through to `None`, so the UI can show "$0.00 (free tier)"
instead of a blank.

### Node state machine + resume

`ResearchCrew.run_with_meta` currently calls each stage method in a hardcoded sequence.
Replace the body with a stage-dispatch loop; each `_run_*_stage` method is unchanged
internally, just invoked through this loop instead of called directly:

```python
STAGES = ["planner", "researcher", "evidence_comparator", "classifier", "writer"]

def run_with_meta(self, topic, force_rerun=False, run_id=None, resume_question_id=None):
    run_id = run_id or str(uuid4())
    start_run(run_id)
    try:
        if resume_question_id is None:
            if not force_rerun:
                reused = self._try_reuse(topic)
                if reused is not None:
                    return reused
            question_id = repository.create_question(topic)
            repository.create_pipeline_run(run_id, question_id, topic)
            stage_outputs = {}
            start_from = 0
        else:
            question_id = resume_question_id
            repository.reattach_pipeline_run(run_id, question_id)
            stage_outputs = self._load_resumable_state(question_id)
            start_from = self._first_unresolved_stage_index(run_id)

        emit_event("run_started", message=f"topic={topic!r}")

        for i, stage in enumerate(STAGES):
            set_stage(stage)
            if i < start_from:
                repository.set_stage_state(run_id, stage, NodeState.SKIPPED_CACHED)
                continue
            repository.set_stage_state(run_id, stage, NodeState.RUNNING)
            emit_event("stage_started")
            try:
                stage_outputs = self._STAGE_RUNNERS[stage](question_id, stage_outputs)
            except PipelineStageError as e:
                repository.set_stage_state(run_id, stage, NodeState.FAILED, detail=e.detail)
                emit_event("stage_failed", detail=e.detail,
                            message=ERROR_SENTENCES[e.error_code][0])
                for remaining in STAGES[i + 1:]:
                    repository.set_stage_state(run_id, remaining, NodeState.NOT_REACHED)
                repository.set_run_status(run_id, "failed")
                raise
            state = NodeState.DEGRADED if stage_outputs.get("_degraded") else NodeState.SUCCEEDED
            repository.set_stage_state(run_id, stage, state)
            emit_event("stage_finished", node_state=state)

        # ... existing conclusion/save/index logic, unchanged ...
        repository.set_run_status(run_id, "succeeded")
        emit_event("run_finished")
        return {...}
    finally:
        clear_run()
```

`stage_outputs` is a small dict threaded between stage runners (`sub_questions`,
`findings`, `contradictions`) — replacing the current local variables — so a resumed run
can seed it from `repository` (sub_questions already on `research_questions`, findings
via `get_findings_list`, contradictions via `get_contradictions`) instead of recomputing
stages that already succeeded. This is why resume is cheap here: every stage's real
output already lives in the traceability DB (per Phase 1/parent-doc principle 3); the
new tables only need to track *which* stage got how far, not duplicate the stage output
itself.

`researcher`'s existing "N/6 sub-questions resolved" outcome maps to `_degraded` in
`stage_outputs` (a boolean the runner sets, not a new return type) → `NodeState.DEGRADED`
per the parent doc's amber/red distinction (§5.3): partial resolution is degraded-but-
continuing, total exhaustion is `PipelineStageError` → failed.

### `run_id` ownership — generated at the API layer, not inside `ResearchCrew`

`run_id` must be known before the pipeline starts executing, not discovered after —
otherwise `/research/async`'s existing `job_id` (generated by
`worker.job_manager.submit()`) and the pipeline's own `run_id` would be two different
strings for the same run, breaking correlation. Fix: `app.py` generates
`run_id = str(uuid4())` itself, in each endpoint, before dispatching:

- `/research/async`: `job_manager.submit()` gains an optional `job_id: str | None`
  parameter — when given, it's used instead of generating a fresh one, so
  `job_id == run_id` and `GET /jobs/{job_id}` and `GET /research/{run_id}` address the
  same run.
- `/research` (synchronous): `run_id` is generated up front, passed into
  `crew.run_with_meta(topic, force_rerun, run_id=run_id)`, and included in
  `ResearchResponse` — the caller only sees it once the whole run finishes (no SSE yet in
  this phase), but it's the same identifier the server logs and `GET /research/{run_id}`
  would answer for mid-run if polled from another connection.
- `ResearchCrew.run_with_meta`'s `run_id: str | None = None` parameter keeps its
  generate-if-absent fallback (`run_id = run_id or str(uuid4())`) only so direct/test
  callers that don't go through `app.py` still work without threading one through.

### New DB tables (`db/models.py`)

```python
class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    run_id = Column(String, primary_key=True)          # uuid4 hex, not autoincrement —
                                                          # generated before question_id exists
    question_id = Column(Integer, ForeignKey("research_questions.id"), nullable=True)
    topic = Column(String, nullable=False)
    status = Column(String, nullable=False, default="running")  # running|succeeded|failed|aborted
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class PipelineStageState(Base):
    __tablename__ = "pipeline_stage_states"
    id = Column(Integer, primary_key=True)
    run_id = Column(String, ForeignKey("pipeline_runs.run_id"), nullable=False)
    stage = Column(String, nullable=False)
    state = Column(String, nullable=False)              # NodeState value
    attempts = Column(JSON, default=list, nullable=False)  # [{deployment, attempt, latency_ms, ...}]
    detail = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
```

Purely additive — no changes to `research_questions`/`sources`/`findings`/
`contradictions`/`conclusions`. `repository.py` gains matching functions
(`create_pipeline_run`, `reattach_pipeline_run`, `set_stage_state`, `set_run_status`,
`get_run`, `get_stage_states`) following the module's existing plain-function-over-a-
session style — no new abstraction layer.

### `POST /research/{run_id}/resume`

```python
@app.post("/research/{run_id}/resume")
async def resume_research(run_id: str):
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Unknown run_id.")
    if run["status"] == "succeeded":
        raise HTTPException(409, "Run already succeeded; nothing to resume.")
    crew = ResearchCrew()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, crew.run_with_meta, run["topic"], False, run_id, run["question_id"]
    )
    return ResearchResponse(...)
```

Reuses the same `run_id` so the client's existing correlation (stdout logs, queue sink)
keeps working across the resume — this is the same run continuing, not a new one.

### `GET /research/{run_id}` (new, small addition not in the parent doc's list but
required for the UI to poll a run's node states without waiting on SSE)

Returns `{run_id, question_id, status, stages: [{stage, state, attempts, detail}]}` from
`repository.get_run` + `get_stage_states`. Cheap and needed by Phase 8 regardless of
whether Phase 3's SSE is live; adding it now avoids a second DB-shape change later.

## Testing

- `tests/test_events.py` — `RunEvent` serialization, `emit_event` no-op with no active
  context, correct `seq` incrementing within a `start_run`/`clear_run` pair.
- `tests/test_errors.py` — every `ErrorCode` has a sentence + severity; `classify_provider_exception`
  covers every exception type `RouterLLM`/`research_crew` currently catch.
- `tests/test_bus.py` — `EventBus.emit` fans out to all sinks; a raising sink doesn't
  stop the others or the caller; `QueueSink` isolates events by `run_id`.
- `tests/test_router_llm_events.py` — extend the existing router_llm tests to assert an
  `llm_attempt` event is emitted per attempt with the right `attempt`/`resolved_deployment`.
- `tests/test_research_crew_stages.py` — the stage-dispatch loop: a mid-pipeline failure
  marks downstream stages `not_reached`; a degraded researcher stage doesn't abort the
  run; resume skips already-succeeded stages and reuses their DB-backed output.
- `tests/test_resume_endpoint.py` — `POST /research/{run_id}/resume` 404s on unknown
  run_id, 409s on an already-succeeded run, and re-enters at the right stage for a
  fixture run with a `failed` stage 3 of 5.

## Open items deferred to explicitly-later phases (not this pass)

- Real SSE consumption of `QueueSink` — Phase 3.
- `NoOpTraceSink` becoming a real Langfuse sink — Phase 7.
- Guardrail-fired events (`guardrail_fired` event_type already reserved in the string
  space above) — Phase 4 wires the emitter, this phase only needs the type to exist as a
  string, not a stub function.
- Cache-hit metrics event payload (tokens/seconds/cost saved) — Phase 5; `cache_hit`
  field on `RunEvent` exists now so Phase 5 doesn't need a schema change, just a call site.
