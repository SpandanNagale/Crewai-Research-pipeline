"""
SQLAlchemy models for the structured persistence / traceability layer.

Traceability chain: Source -> Finding -> Conclusion, with Contradiction
linking two findings. Every Finding carries the source_id it came from;
every Conclusion carries the finding_ids it was derived from, so a
conclusion can always be walked back to the raw sources that support it.
"""
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ResearchQuestion(Base):
    __tablename__ = "research_questions"

    id = Column(Integer, primary_key=True)
    topic = Column(String, nullable=False)
    sub_questions = Column(JSON, default=list, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    sources = relationship("Source", back_populates="question", cascade="all, delete-orphan")
    findings = relationship("Finding", back_populates="question", cascade="all, delete-orphan")
    conclusions = relationship("Conclusion", back_populates="question", cascade="all, delete-orphan")


class Source(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey("research_questions.id"), nullable=False)
    url = Column(String, nullable=False)
    title = Column(String)
    retrieved_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    raw_content = Column(Text)

    question = relationship("ResearchQuestion", back_populates="sources")
    findings = relationship("Finding", back_populates="source", cascade="all, delete-orphan")


class Finding(Base):
    __tablename__ = "findings"

    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("research_questions.id"), nullable=False)
    text = Column(Text, nullable=False)
    category = Column(String, nullable=True)
    # Not in the original spec's column list, but required for the
    # EvidenceComparator to know which findings answer the same sub-question,
    # and for the Writer to group findings by sub-question when reading from
    # the DB. Kept as a plain string rather than a new table to stay minimal.
    sub_question = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    source = relationship("Source", back_populates="findings")
    question = relationship("ResearchQuestion", back_populates="findings")


class Contradiction(Base):
    __tablename__ = "contradictions"

    id = Column(Integer, primary_key=True)
    finding_a_id = Column(Integer, ForeignKey("findings.id"), nullable=False)
    finding_b_id = Column(Integer, ForeignKey("findings.id"), nullable=False)
    explanation = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Conclusion(Base):
    __tablename__ = "conclusions"

    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey("research_questions.id"), nullable=False)
    text = Column(Text, nullable=False)
    finding_ids = Column(JSON, default=list, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    question = relationship("ResearchQuestion", back_populates="conclusions")


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


class ClaimGrounding(Base):
    __tablename__ = "claim_groundings"

    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey("research_questions.id"), nullable=False)
    claim_text = Column(Text, nullable=False)
    grounded = Column(Boolean, nullable=False)
    supporting_finding_id = Column(Integer, ForeignKey("findings.id"), nullable=True)
    objection = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class SearchCacheEntry(Base):
    """
    TTL-bounded cache of search_tool results, keyed on the normalised query
    (see tools/search_tool.py::_cache_key). latency_ms is the real wall-clock
    time the original provider call took, replayed as the "seconds saved"
    metric on a cache hit rather than a fabricated estimate.
    """
    __tablename__ = "search_cache_entries"

    query_key = Column(String, primary_key=True)
    results_json = Column(JSON, nullable=False)
    latency_ms = Column(Float, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Job(Base):
    """
    Status of a POST /research/async job (defect #14: this used to live only
    in worker/job_manager.py's in-memory dict, so a process restart silently
    lost every in-flight/completed job's status).
    """
    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True)
    status = Column(String, nullable=False, default="pending")  # pending|running|done|failed
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class BudgetEntry(Base):
    """
    Append-only ledger of real LLM spend (RunEvent.cost_usd on every
    llm_attempt, see observability/budget.py::BudgetSink) -- summed over
    "today" to enforce settings.DAILY_BUDGET_USD's hard kill-switch. Kept
    granular (one row per attempt) rather than a running counter so it can't
    drift and needs no read-modify-write locking.
    """
    __tablename__ = "budget_entries"

    id = Column(Integer, primary_key=True)
    amount_usd = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
