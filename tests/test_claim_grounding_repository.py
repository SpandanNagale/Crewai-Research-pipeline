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


def test_replace_claim_groundings_round_trip():
    repository.create_pipeline_run("run-1", question_id=1, topic="x")
    repository.replace_claim_groundings(1, [
        {"claim_text": "A", "grounded": True, "supporting_finding_id": 5, "objection": None},
        {"claim_text": "B", "grounded": False, "supporting_finding_id": None, "objection": "no support"},
    ])
    rows = repository.get_claim_groundings(1)
    assert len(rows) == 2
    assert rows[0]["claim_text"] == "A"
    assert rows[0]["grounded"] is True
    assert rows[1]["objection"] == "no support"


def test_replace_claim_groundings_clears_prior_set():
    repository.replace_claim_groundings(1, [{"claim_text": "old", "grounded": True}])
    repository.replace_claim_groundings(1, [{"claim_text": "new", "grounded": False, "objection": "x"}])
    rows = repository.get_claim_groundings(1)
    assert len(rows) == 1
    assert rows[0]["claim_text"] == "new"
