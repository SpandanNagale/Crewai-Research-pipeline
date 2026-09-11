from typing import Dict, List

from crewai import Agent, Task


def _group_by_subquestion(findings: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for f in findings:
        key = f.get("sub_question") or "General"
        grouped.setdefault(key, []).append(f)
    return grouped


def get_write_task(
    writer: Agent,
    topic: str,
    findings: List[dict],
    contradictions: List[dict],
    unresolved_sub_questions: List[str] = None,
    feedback: List[str] = None,
) -> Task:
    grouped = _group_by_subquestion(findings)

    section_blocks = []
    for sub_question, items in grouped.items():
        lines = "\n".join(
            f"  [{it['id']}] ({it.get('category') or 'uncategorized'}) {it['text']} "
            f"— Source: {it.get('source_title') or 'unknown'} ({it.get('source_url') or 'n/a'})"
            for it in items
        )
        section_blocks.append(f"### {sub_question}\n{lines}")
    findings_block = "\n\n".join(section_blocks) if section_blocks else "No findings recorded."

    if contradictions:
        contradictions_block = "\n".join(
            f"- Finding [{c['finding_a_id']}] vs Finding [{c['finding_b_id']}]: {c['explanation']}"
            for c in contradictions
        )
    else:
        contradictions_block = "None detected."

    unresolved_sub_questions = unresolved_sub_questions or []
    if unresolved_sub_questions:
        unresolved_block = (
            "\n\nThe following sub-questions could NOT be resolved (no findings were "
            "extracted for them after retrying) — explicitly list them in a "
            "'Unresolved Questions' subsection under Findings. Do NOT fabricate "
            "content for them:\n" + "\n".join(f"- {q}" for q in unresolved_sub_questions)
        )
    else:
        unresolved_block = ""

    feedback = feedback or []
    if feedback:
        feedback_block = (
            "\n\nA prior draft of this report was reviewed and found to contain "
            "unsupported claims. Fix these specific issues in this draft — either "
            "ground the claim in a listed finding or remove it:\n"
            + "\n".join(f"- {f}" for f in feedback)
        )
    else:
        feedback_block = ""

    return Task(
        description=(
            f'Write a complete markdown research report on: "{topic}"\n\n'
            "You must synthesize ONLY from the findings listed below — every "
            "finding was already extracted from a source and stored in the "
            "database, do not invent new facts or sources. Each finding shows "
            "its database id, category, and source.\n\n"
            f"FINDINGS (grouped by sub-question):\n{findings_block}{unresolved_block}\n\n"
            f"FLAGGED CONTRADICTIONS between findings (from the "
            f"EvidenceComparator stage):\n{contradictions_block}"
            f"{feedback_block}\n\n"
            "Structure the report as:\n"
            "1. **Executive Summary** (3-5 sentences)\n"
            "2. **Findings** (one section per sub-question, citing sources)\n"
            "3. **Contradictions & Disagreements** — explicitly mention EVERY "
            "flagged contradiction above and explain both sides. Do NOT "
            "silently resolve a contradiction by picking one side, and do NOT "
            "omit it. Write 'None detected.' only if the list above is empty.\n"
            "4. **Conclusion** (key takeaways and open questions)\n"
            "5. **References** (all sources cited, numbered)\n\n"
            "Write for a technical but non-specialist audience. Be concise and "
            "avoid filler sentences."
        ),
        expected_output=(
            "A complete markdown document with all five sections, including an "
            "explicit Contradictions & Disagreements section."
        ),
        agent=writer,
    )
