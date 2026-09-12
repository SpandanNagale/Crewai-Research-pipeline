"""
Phase 7: Langfuse tracing. Consumes the same RunEvents every other sink
sees (see observability/bus.py) and turns them into one Langfuse trace per
run_id, with one nested span per pipeline stage and a child observation for
every llm_attempt/cache_hit/guardrail_fired.

Two things are deliberately kept independent of any live sink/process:

- trace_id is *derived* from run_id via Langfuse.create_trace_id(seed=...),
  a pure SHA-256 hash -- no client, no network call, no shared state
  needed. That means app.py (building the API response) and eval/harness.py
  (a separate process scoring a completed run against a *deployed*
  instance) can both recompute the exact same trace_id from a run_id alone,
  without coordinating with whichever process actually ran the pipeline.
- Building a Langfuse() client with missing/invalid keys degrades to a
  no-op tracer internally (never raises), but a few of the convenience
  methods we use (get_trace_url, create_score) still do real network I/O
  and can raise on their own. Every public function here is wrapped so a
  Langfuse outage degrades to "no trace URL" / "score not recorded",
  never a broken pipeline run or API response -- same contract as every
  other Sink (see EventBus.emit's try/except).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from opentelemetry import trace as otel_trace_api

from config.settings import settings
from observability.events import RunEvent

logger = logging.getLogger(__name__)

_client_lock = threading.Lock()
_client: Optional[Any] = None


def is_configured() -> bool:
    return bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)


def _get_client():
    """Lazy, process-wide singleton so the SDK's own batching/flush thread is reused."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from langfuse import Langfuse

                _client = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST,
                )
    return _client


def trace_id_for(run_id: str) -> str:
    from langfuse import Langfuse

    return Langfuse.create_trace_id(seed=run_id)


def get_trace_url(run_id: str) -> Optional[str]:
    """Public trace URL for the API response's "view trace" link. None if unconfigured or unreachable."""
    if not is_configured():
        return None
    try:
        return _get_client().get_trace_url(trace_id=trace_id_for(run_id))
    except Exception:
        logger.warning("Failed to build Langfuse trace URL for run %s", run_id, exc_info=True)
        return None


def attach_score(run_id: str, name: str, value: float, comment: Optional[str] = None) -> None:
    """Attach a NUMERIC score to a run's trace -- used by the eval harness for metrics
    (e.g. citation_precision) that are only known after the run completes, outside the
    pipeline process itself."""
    if not is_configured():
        return
    try:
        client = _get_client()
        client.create_score(
            trace_id=trace_id_for(run_id), name=name, value=value,
            data_type="NUMERIC", comment=comment,
        )
        client.flush()
    except Exception:
        logger.warning("Failed to attach Langfuse score %s for run %s", name, run_id, exc_info=True)


_STAGE_OBSERVATION_TYPES = {
    "guardrail_fired": "guardrail",
}


def _start_nested(parent, *, name: str, as_type: str = "span", **kwargs):
    """Create a child observation of `parent` via real OTel context propagation
    (not the trace_context={"trace_id":...} remote-parent path -- that path marks
    *every* observation it touches AS_ROOT, which made the trace's own name/input
    flicker to whatever child happened to be processed last server-side)."""
    with otel_trace_api.use_span(parent._otel_span, end_on_exit=False):
        return _get_client().start_observation(name=name, as_type=as_type, **kwargs)


def _event_nested(parent, *, name: str, **kwargs) -> None:
    with otel_trace_api.use_span(parent._otel_span, end_on_exit=False):
        _get_client().create_event(name=name, **kwargs)


class LangfuseSink:
    """EventBus Sink: one trace per run_id, one span per stage, nested child
    observations for llm_attempt/cache_hit/guardrail_fired. See module docstring."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # run_id -> {"trace_id", "root": obj, "stages": {stage_name: obj}}
        self._runs: Dict[str, Dict[str, Any]] = {}

    def _run_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._runs.get(run_id)

    def _current_parent(self, state: Dict[str, Any], stage: str):
        return state["stages"].get(stage) or state["root"]

    def emit(self, event: RunEvent) -> None:
        try:
            self._emit(event)
        except Exception:
            logger.warning("LangfuseSink failed to process event %s", event.event_type, exc_info=True)

    def _emit(self, event: RunEvent) -> None:
        if event.event_type == "run_started":
            self._on_run_started(event)
            return

        state = self._run_state(event.run_id)
        if state is None:
            return  # trace couldn't be created (e.g. client misconfigured mid-run)

        if event.event_type == "stage_started":
            self._on_stage_started(event, state)
        elif event.event_type == "stage_finished":
            self._on_stage_finished(event, state)
        elif event.event_type == "stage_failed":
            self._on_stage_failed(event, state)
        elif event.event_type == "llm_attempt":
            self._on_llm_attempt(event, state)
        elif event.event_type == "cache_hit":
            self._on_generic_child(event, state, name="cache_hit")
        elif event.event_type == "guardrail_fired":
            self._on_generic_child(event, state, name="guardrail_fired")
        elif event.event_type == "run_finished":
            self._on_run_finished(event, state)

        if event.score_name is not None and event.score_value is not None:
            attach_score(event.run_id, event.score_name, event.score_value)

    def _on_run_started(self, event: RunEvent) -> None:
        client = _get_client()
        trace_id = trace_id_for(event.run_id)
        root = client.start_observation(
            trace_context={"trace_id": trace_id, "parent_span_id": None},
            name="research_run", as_type="span",
            input={"message": event.message},
            metadata={"run_id": event.run_id},
        )
        with self._lock:
            self._runs[event.run_id] = {"trace_id": trace_id, "root": root, "stages": {}}

    def _on_stage_started(self, event: RunEvent, state: Dict[str, Any]) -> None:
        span = _start_nested(state["root"], name=event.stage, as_type="span")
        state["stages"][event.stage] = span

    def _on_stage_finished(self, event: RunEvent, state: Dict[str, Any]) -> None:
        span = state["stages"].pop(event.stage, None)
        if span is not None:
            span.update(output={"node_state": event.node_state.value if event.node_state else None})
            span.end()

    def _on_stage_failed(self, event: RunEvent, state: Dict[str, Any]) -> None:
        span = state["stages"].get(event.stage)
        if span is not None:
            span.update(level="ERROR", status_message=event.detail)

    def _on_llm_attempt(self, event: RunEvent, state: Dict[str, Any]) -> None:
        parent = self._current_parent(state, event.stage)
        usage = {}
        if event.prompt_tokens is not None:
            usage["input"] = event.prompt_tokens
        if event.completion_tokens is not None:
            usage["output"] = event.completion_tokens
        cost = {"total": event.cost_usd} if event.cost_usd is not None else None
        level = "DEFAULT" if event.message in ("ok", "cache_hit") else "WARNING"
        gen = _start_nested(
            parent, name="llm_call", as_type="generation",
            model=event.resolved_deployment,
            usage_details=usage or None,
            cost_details=cost,
            level=level,
            status_message=event.message,
            metadata={
                "attempt": event.attempt, "cache_hit": event.cache_hit,
                "latency_ms": event.latency_ms, "detail": event.detail,
            },
        )
        gen.end()

    def _on_generic_child(self, event: RunEvent, state: Dict[str, Any], name: str) -> None:
        parent = self._current_parent(state, event.stage)
        as_type = _STAGE_OBSERVATION_TYPES.get(event.event_type)
        metadata = {
            "message": event.message, "detail": event.detail,
            "tokens_saved": event.tokens_saved, "cost_avoided_usd": event.cost_avoided_usd,
            "seconds_saved": event.seconds_saved,
        }
        if as_type is not None:
            obs = _start_nested(parent, name=name, as_type=as_type, metadata=metadata)
            obs.end()
        else:
            _event_nested(parent, name=name, metadata=metadata)

    def _on_run_finished(self, event: RunEvent, state: Dict[str, Any]) -> None:
        state["root"].update(output={"message": event.message})
        state["root"].end()
        with self._lock:
            self._runs.pop(event.run_id, None)
        _get_client().flush()
