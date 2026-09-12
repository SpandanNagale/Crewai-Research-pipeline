from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import db.session as db_session
from config.settings import settings
from db import repository
from db.models import Base
from observability import budget
from observability.events import RunEvent


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    test_session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_session, "SessionLocal", test_session_local)
    yield


def _llm_attempt(cost_usd, stage="planner"):
    return RunEvent(
        run_id="r1", seq=1, timestamp=datetime.utcnow(), stage=stage,
        event_type="llm_attempt", cost_usd=cost_usd, message="ok",
    )


def test_is_exceeded_false_when_budget_disabled(monkeypatch):
    monkeypatch.setattr(settings, "DAILY_BUDGET_USD", 0)
    repository.record_spend(1000.0)
    assert budget.is_exceeded() is False
    assert budget.remaining_usd() is None


def test_is_exceeded_false_below_budget(monkeypatch):
    monkeypatch.setattr(settings, "DAILY_BUDGET_USD", 5.0)
    repository.record_spend(1.0)
    repository.record_spend(2.0)
    assert budget.spent_today_usd() == pytest.approx(3.0)
    assert budget.is_exceeded() is False
    assert budget.remaining_usd() == pytest.approx(2.0)


def test_is_exceeded_true_at_or_above_budget(monkeypatch):
    monkeypatch.setattr(settings, "DAILY_BUDGET_USD", 5.0)
    repository.record_spend(5.0)
    assert budget.is_exceeded() is True
    assert budget.remaining_usd() == 0.0


def test_get_spend_since_ignores_entries_before_cutoff():
    with db_session.get_session() as session:
        from db.models import BudgetEntry
        old = BudgetEntry(amount_usd=100.0)
        session.add(old)
        session.flush()
        old.created_at = datetime.utcnow() - timedelta(days=2)

    repository.record_spend(1.5)
    assert repository.get_spend_since(datetime.utcnow() - timedelta(hours=1)) == pytest.approx(1.5)


def test_budget_sink_records_cost_on_llm_attempt():
    sink = budget.BudgetSink()
    sink.emit(_llm_attempt(0.002))
    sink.emit(_llm_attempt(0.003))
    assert budget.spent_today_usd() == pytest.approx(0.005)


def test_budget_sink_ignores_non_llm_attempt_events():
    sink = budget.BudgetSink()
    sink.emit(RunEvent(run_id="r1", seq=1, timestamp=datetime.utcnow(), stage="planner", event_type="stage_started"))
    assert budget.spent_today_usd() == 0.0


def test_budget_sink_ignores_llm_attempts_with_no_cost():
    sink = budget.BudgetSink()
    sink.emit(_llm_attempt(None))
    assert budget.spent_today_usd() == 0.0


def test_budget_sink_swallows_db_errors(monkeypatch):
    def boom(amount):
        raise RuntimeError("db down")

    monkeypatch.setattr(repository, "record_spend", boom)
    sink = budget.BudgetSink()
    sink.emit(_llm_attempt(0.01))  # must not raise
