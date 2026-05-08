import yaml
from crewai import Agent
from agents.base import get_llm

def get_critic() -> Agent:
    with open("config/agents_config.yaml") as f:
        cfg = yaml.safe_load(f)["critic"]

    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )
