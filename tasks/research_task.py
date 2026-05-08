from crewai import Task, Agent

def get_research_task(researcher: Agent, plan_task: Task) -> Task:
    return Task(
        description=(
            "Using the research plan provided, answer each sub-question thoroughly. "
            "For each question:\n"
            "1. Search the web and/or Wikipedia for relevant information.\n"
            "2. Summarise the key findings in 2–4 sentences.\n"
            "3. Include the source URL or reference.\n\n"
            "Be factual. If information is unavailable or conflicting, say so explicitly."
        ),
        expected_output=(
            "A structured findings document with each sub-question as a heading, "
            "followed by a factual answer and at least one source citation."
        ),
        agent=researcher,
        context=[plan_task],
    )
