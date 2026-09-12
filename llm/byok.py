"""
Phase 9: an optional bring-your-own-key override, so a caller who has hit
the deployment's daily budget kill-switch (observability/budget.py) can keep
going on their own dime instead of being locked out entirely.

Thread-local, same mechanism as observability/context.py's RunContext --
each pipeline run executes synchronously on one worker thread, so setting
this at the top of that thread's run and clearing it in a finally block
gives every get_llm() call on that thread ambient access to the override
without threading a parameter through every agent/task/crew call site.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

_local = threading.local()


@dataclass(frozen=True)
class BYOKOverride:
    provider: str
    api_key: str
    model: str


def set_byok(override: Optional[BYOKOverride]) -> None:
    _local.override = override


def get_byok() -> Optional[BYOKOverride]:
    return getattr(_local, "override", None)


def clear_byok() -> None:
    _local.override = None
