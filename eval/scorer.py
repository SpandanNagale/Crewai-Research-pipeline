import re
from dataclasses import dataclass
from typing import Optional

from llm.provider import get_llm


@dataclass
class EvalResult:
    citation_score: Optional[float]   # 0-10: how well claims are cited
    coverage_score: Optional[float]   # 0-10: how completely the topic is covered
    clarity_score: Optional[float]    # 0-10: readability and structure
    total: Optional[float]            # weighted average; None if any dimension failed to parse
    feedback: str                     # short critique from the LLM judge
    parse_failed: bool = False        # defect #8: a parse failure is never silently a 0.0

    def __str__(self) -> str:
        if self.parse_failed:
            return f"Judge output could not be parsed.\n\nFeedback:\n{self.feedback}"
        return (
            f"Citation score : {self.citation_score:.1f}/10\n"
            f"Coverage score : {self.coverage_score:.1f}/10\n"
            f"Clarity score  : {self.clarity_score:.1f}/10\n"
            f"Total          : {self.total:.1f}/10\n\n"
            f"Feedback:\n{self.feedback}"
        )


def score_report(report: str, topic: str) -> EvalResult:
    """
    Use an LLM judge to score the research report on 3 dimensions. Uses the
    "evaluator" entry in config/agents_config.yaml if present (falls back to
    the global LLM_PROVIDER/LLM_MODEL), so scoring can run on a different
    provider/model than the pipeline agents.
    Returns an EvalResult with numeric scores and written feedback.

    Defect #8 fix: a judge response that doesn't parse is never silently
    scored as 0.0 (indistinguishable from "the judge gave this report the
    worst possible score"). parse_failed is set instead, and any dimension
    that couldn't be parsed comes back None -- total is None whenever any
    dimension failed to parse, since a weighted average with a missing term
    isn't a real average.
    """
    llm = get_llm("evaluator")

    prompt = f"""You are a strict academic reviewer. Evaluate the following research report on the topic "{topic}".

Score it on three dimensions from 0 to 10:
1. Citation score — Are claims backed by sources? Are sources listed in a References section?
2. Coverage score — Does the report fully address the topic with enough depth and breadth?
3. Clarity score  — Is the report well-structured, readable, and free of vague language?

Respond in this EXACT format (numbers only, no extra text on score lines):
CITATION: <score>
COVERAGE: <score>
CLARITY: <score>
FEEDBACK: <2-3 sentences of constructive critique>

--- REPORT START ---
{report[:6000]}
--- REPORT END ---
"""

    text = llm.call([{"role": "user", "content": prompt}])

    citation = _extract_score(text, "CITATION")
    coverage = _extract_score(text, "COVERAGE")
    clarity = _extract_score(text, "CLARITY")
    feedback = _extract_feedback(text)

    parse_failed = citation is None or coverage is None or clarity is None
    total = None if parse_failed else round((citation * 0.4 + coverage * 0.35 + clarity * 0.25), 2)

    return EvalResult(
        citation_score=citation,
        coverage_score=coverage,
        clarity_score=clarity,
        total=total,
        feedback=feedback,
        parse_failed=parse_failed,
    )


def _extract_score(text: str, label: str) -> Optional[float]:
    match = re.search(rf"{label}:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        return min(10.0, float(match.group(1)))
    return None


def _extract_feedback(text: str) -> str:
    match = re.search(r"FEEDBACK:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No feedback extracted."
