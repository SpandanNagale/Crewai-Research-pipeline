from crewai import Task, Agent

def get_critique_task(critic: Agent, write_task: Task) -> Task:
    return Task(
        description=(
            "Review the draft report carefully. Check for:\n"
            "- Claims made without a source citation\n"
            "- Logical inconsistencies or contradictions\n"
            "- Sections that are vague, repetitive, or unclear\n"
            "- Missing context a reader would need\n\n"
            "Fix all issues directly in the report and return the improved version in full. "
            "Do not just list problems — produce the corrected markdown document."
        ),
        expected_output=(
            "The complete, corrected markdown report. "
            "Same structure as the draft but with all issues resolved."
        ),
        agent=critic,
        context=[write_task],
    )
