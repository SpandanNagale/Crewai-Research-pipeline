from crewai import Crew, Process
from agents.planner import get_planner
from agents.researcher import get_researcher
from agents.writer import get_writer
from agents.critic import get_critic
from tasks.plan_task import get_plan_task
from tasks.research_task import get_research_task
from tasks.write_task import get_write_task
from tasks.critique_task import get_critique_task
from tools.file_tool import save_report


class ResearchCrew:
    def run(self, topic: str) -> str:
        planner    = get_planner()
        researcher = get_researcher()
        writer     = get_writer()
        critic     = get_critic()

        plan_task     = get_plan_task(topic, planner)
        research_task = get_research_task(researcher, plan_task)
        write_task    = get_write_task(writer, research_task)
        critique_task = get_critique_task(critic, write_task)

        crew = Crew(
            agents=[planner, researcher, writer, critic],
            tasks=[plan_task, research_task, write_task, critique_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        filepath = save_report(str(result), topic)
        print(f"\nReport saved to: {filepath}")
        return str(result)
