"""Pydantic models used as `output_pydantic` schemas for structured task outputs."""
from typing import List

from pydantic import BaseModel, Field

FINDING_CATEGORIES = ["adoption", "technology", "risk", "regulatory", "case-study", "other"]


class FindingItem(BaseModel):
    finding_text: str = Field(description="A short, atomic, single-claim finding (under 20 words).")
    source_title: str = Field(description="Title of the source the finding came from.")
    source_url: str = Field(description="URL of the source the finding came from.")


class ResearchOutput(BaseModel):
    """One sub-question's worth of findings — the Researcher stage runs once per sub-question."""

    findings: List[FindingItem] = Field(default_factory=list)


class ContradictionItem(BaseModel):
    finding_a_id: int
    finding_b_id: int
    explanation: str = Field(description="What the two findings disagree about.")


class ComparisonOutput(BaseModel):
    contradictions: List[ContradictionItem] = Field(default_factory=list)


class ClassificationItem(BaseModel):
    finding_id: int
    category: str = Field(description=f"One of: {', '.join(FINDING_CATEGORIES)}")


class ClassificationOutput(BaseModel):
    items: List[ClassificationItem] = Field(default_factory=list)
