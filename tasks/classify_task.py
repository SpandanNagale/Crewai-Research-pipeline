from typing import List

from crewai import Agent, Task

from tasks.schemas import FINDING_CATEGORIES

JSON_SHAPE = '{"items": [{"finding_id": 1, "category": "..."}]}'


def get_classify_task(classifier: Agent, findings: List[dict]) -> Task:
    lines = "\n".join(f"- finding_id={f['id']}: {f['text']}" for f in findings)
    categories = ", ".join(FINDING_CATEGORIES)

    return Task(
        description=(
            f"Classify each finding below into exactly one category from this "
            f"fixed list: {categories}.\n\n"
            f"Findings:\n{lines}\n\n"
            "If nothing fits well, use 'other'.\n\n"
            "Respond with ONLY a single raw JSON object — no markdown code "
            f"fences, no commentary — in exactly this shape:\n{JSON_SHAPE}\n"
            "Cover every finding_id given above exactly once."
        ),
        expected_output=(
            f"Only the raw JSON object in this exact shape, nothing else: {JSON_SHAPE}"
        ),
        agent=classifier,
    )
