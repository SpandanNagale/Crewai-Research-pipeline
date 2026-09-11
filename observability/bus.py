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
