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
