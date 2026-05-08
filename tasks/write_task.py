from crewai import Task, Agent

def get_write_task(writer: Agent, research_task: Task) -> Task:
    return Task(
        description=(
            "Using the research findings provided, write a complete markdown report. "
            "Structure it as follows:\n\n"
            "1. **Executive Summary** (3–5 sentences overview)\n"
            "2. **Findings** (one section per sub-question, with a clear heading)\n"
            "3. **Conclusion** (key takeaways and open questions)\n"
            "4. **References** (all sources cited in the findings)\n\n"
            "Write for a technical but non-specialist audience. "
            "Be concise and avoid filler sentences."
        ),
        expected_output=(
            "A complete markdown document with all four sections. "
            "Headings use ## and ###. References use numbered footnote style."
        ),
        agent=writer,
        context=[research_task],
    )
