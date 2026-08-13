from typing import Dict, List

from crewai import Agent, Task

JSON_SHAPE = (
    '{"contradictions": [{"finding_a_id": 1, "finding_b_id": 2, "explanation": "..."}]}'
)


def get_compare_task(comparator: Agent, findings_by_subquestion: Dict[str, List[dict]]) -> Task:
    blocks = []
    for sub_question, findings in findings_by_subquestion.items():
        lines = "\n".join(
            f"  - finding_id={f['id']}: {f['text']} (source: {f['source_url']})"
            for f in findings
        )
        blocks.append(f"Sub-question: {sub_question}\n{lines}")
    body = "\n\n".join(blocks) if blocks else "No findings to compare."

    return Task(
        description=(
            "Below, findings are grouped by the sub-question they answer. For "
            "each group, compare the findings against each other and identify "
            "pairs that genuinely DISAGREE or CONTRADICT one another — e.g. "
            "different numbers, opposite claims, or conflicting conclusions.\n\n"
            "Do NOT flag findings that are merely complementary, or that simply "
            "add different information without conflicting. Only flag true "
            "contradictions. Keep each explanation under 20 words.\n\n"
            f"{body}\n\n"
            "For every contradiction, report the two finding_ids involved and a "
            "short explanation of exactly what they disagree about.\n\n"
            "Respond with ONLY a single raw JSON object — no markdown code "
            f"fences, no commentary — in exactly this shape:\n{JSON_SHAPE}\n"
            "Use an empty 'contradictions' list if none are found."
        ),
        expected_output=(
            f"Only the raw JSON object in this exact shape, nothing else: {JSON_SHAPE}"
        ),
        agent=comparator,
    )
