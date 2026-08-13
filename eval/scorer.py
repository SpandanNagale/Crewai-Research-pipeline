import re
from dataclasses import dataclass
from llm.provider import get_llm


@dataclass
class EvalResult:
    citation_score: float     # 0–10: how well claims are cited
    coverage_score: float     # 0–10: how completely the topic is covered
    clarity_score: float      # 0–10: readability and structure
    total: float              # weighted average
    feedback: str             # short critique from the LLM judge

    def __str__(self) -> str:
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
    clarity  = _extract_score(text, "CLARITY")
    feedback = _extract_feedback(text)

    total = round((citation * 0.4 + coverage * 0.35 + clarity * 0.25), 2)

    return EvalResult(
        citation_score=citation,
        coverage_score=coverage,
        clarity_score=clarity,
        total=total,
        feedback=feedback,
    )


def _extract_score(text: str, label: str) -> float:
    match = re.search(rf"{label}:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        return min(10.0, float(match.group(1)))
    return 0.0


def _extract_feedback(text: str) -> str:
    match = re.search(r"FEEDBACK:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No feedback extracted."
