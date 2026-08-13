from crewai import Agent, Task

JSON_SHAPE = (
    '{"findings": [{"finding_text": "...", "source_title": "...", "source_url": "..."}]}'
)


def get_research_task(researcher: Agent, sub_question: str) -> Task:
    return Task(
        description=(
            f'Answer this single research sub-question: "{sub_question}"\n\n'
            "Search the web and/or Wikipedia. Do not write a summary paragraph — "
            "instead, extract 1-2 discrete, atomic findings: each finding is one "
            "short factual claim (under 20 words) tied to exactly one source.\n\n"
            "Rules:\n"
            "- Exactly 1-2 findings, ideally from different sources.\n"
            "- If no evidence can be found, return exactly one finding "
            "explaining that, with source_title 'None' and source_url 'none'.\n"
            "- Be factual. Do not speculate or invent sources.\n\n"
            "Once you have your findings, respond with ONLY a single raw JSON "
            "object — no markdown code fences, no commentary before or after "
            f"it — in exactly this shape:\n{JSON_SHAPE}"
        ),
        expected_output=(
            f"Only the raw JSON object in this exact shape, nothing else: {JSON_SHAPE}"
        ),
        agent=researcher,
    )
