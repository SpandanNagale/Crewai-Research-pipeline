import yaml
from crewai import Agent
from agents.base import get_llm
from tools.search_tool import get_search_tool
from tools.wiki_tool import get_wiki_tool

def get_researcher() -> Agent:
    with open("config/agents_config.yaml") as f:
        cfg = yaml.safe_load(f)["researcher"]

    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=get_llm("researcher"),
        tools=[get_search_tool(), get_wiki_tool()],
        verbose=True,
        allow_delegation=False,
        max_iter=15,            # prevent infinite tool loops
    )
