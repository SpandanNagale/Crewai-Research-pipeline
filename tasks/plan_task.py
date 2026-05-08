from crewai import Task
from crewai import Agent

def get_plan_task(topic: str, planner: Agent) -> Task:
    return Task(
        description=(
            f"Create a research plan for the following topic:\n\n"
            f"**Topic:** {topic}\n\n"
            f"Produce 4–6 specific, focused sub-questions that together cover "
            f"the topic comprehensively. Each sub-question should be answerable "
            f"with a web search or Wikipedia lookup."
        ),
        expected_output=(
            "A numbered list of 4–6 specific sub-questions, each on its own line. "
            "No preamble, no explanations — just the questions."
        ),
        agent=planner,
    )
