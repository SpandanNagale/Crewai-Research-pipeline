"""
Phase 9: a hard daily spend kill-switch across the whole deployment.

BudgetSink is a plain EventBus Sink (see observability/bus.py) -- it just
records every real llm_attempt's cost_usd to the `budget_entries` table
(db/repository.py::record_spend), the same "sinks consume RunEvents" pattern
StdoutSink/QueueSink/LangfuseSink already use. Spend is recorded
unconditionally (even with settings.DAILY_BUDGET_USD unset) so a cap turned
on later has real history to check against, and so the true number is always
available -- never fabricated.

is_exceeded()/remaining_usd() are what app.py's endpoints check before
starting a new pipeline run. A BYOK request (see llm/byok.py) bypasses this
check entirely -- the user is spending their own money, not the
deployment's.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from config.settings import settings
from observability.events import RunEvent

logger = logging.getLogger(__name__)


def _start_of_today_utc() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)


def spent_today_usd() -> float:
    from db import repository

    return repository.get_spend_since(_start_of_today_utc())


def is_exceeded() -> bool:
    if settings.DAILY_BUDGET_USD <= 0:
        return False
    return spent_today_usd() >= settings.DAILY_BUDGET_USD


def remaining_usd() -> Optional[float]:
    if settings.DAILY_BUDGET_USD <= 0:
        return None
    return max(0.0, settings.DAILY_BUDGET_USD - spent_today_usd())


class BudgetSink:
    def emit(self, event: RunEvent) -> None:
        if event.event_type != "llm_attempt" or not event.cost_usd:
            return
        try:
            from db import repository

            repository.record_spend(event.cost_usd)
        except Exception:
            logger.warning("BudgetSink failed to record spend", exc_info=True)
