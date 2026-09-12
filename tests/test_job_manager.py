import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import db.session as db_session
from db.models import Base
from worker import job_manager


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    # StaticPool: submit() runs fn() on a ThreadPoolExecutor thread, not
    # pytest's -- see tests/test_run_endpoints.py's isolated_db fixture for
    # why plain check_same_thread=False isn't enough for an in-memory sqlite.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    test_session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_session, "SessionLocal", test_session_local)
    yield


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


def test_get_job_returns_none_for_unknown_job_id():
    assert job_manager.get_job("no-such-job") is None


def test_failed_job_records_the_error():
    def boom():
        raise ValueError("kaboom")

    job_id = job_manager.submit(boom, job_id="failing-job")
    for _ in range(50):
        job = job_manager.get_job(job_id)
        if job["status"] == "failed":
            break
        time.sleep(0.02)
    assert job["status"] == "failed"
    assert "kaboom" in job["error"]
