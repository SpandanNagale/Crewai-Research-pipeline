"""
SQLAlchemy models for the structured persistence / traceability layer.

Traceability chain: Source -> Finding -> Conclusion, with Contradiction
linking two findings. Every Finding carries the source_id it came from;
every Conclusion carries the finding_ids it was derived from, so a
conclusion can always be walked back to the raw sources that support it.
"""
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text
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
