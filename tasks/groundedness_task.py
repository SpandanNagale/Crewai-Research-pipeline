from typing import List

from crewai import Agent, Task


def get_groundedness_task(critic: Agent, report: str, findings: List[dict]) -> Task:
    findings_block = "\n".join(
        f"[{f['id']}] {f['text']} — Source: {f.get('source_title') or 'unknown'} ({f.get('source_url') or 'n/a'})"
        for f in findings
    ) or "No findings recorded."

    return Task(
        description=(
            "Check a research report for groundedness: every factual claim in the "
            "report must be supported by one of the findings listed below (each has "
            "a database id). Identify each distinct factual claim in the report, and "
            "for each one say whether it is supported by a listed finding.\n\n"
            f"FINDINGS (the only allowed evidence):\n{findings_block}\n\n"
            "--- REPORT TO CHECK START ---\n"
            f"{report}\n"
            "--- REPORT TO CHECK END ---\n\n"
            "Respond with ONLY a JSON object matching this shape, no prose outside "
            "the JSON:\n"
            '{"overall_score": <0.0-1.0, fraction of claims that are grounded>, '
            '"claims": [{"claim_text": "...", "grounded": true/false, '
            '"supporting_finding_id": <id or null>, "objection": "<why not '
            'grounded, or null>"}]}'
        ),
        expected_output="A single JSON object with overall_score and a claims array, nothing else.",
        agent=critic,
    )
