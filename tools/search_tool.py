from crewai.tools import tool
import os
import requests
from config.settings import settings

os.environ["SERPER_API_KEY"] = settings.SERPER_API_KEY

@tool("Web Search")
def search_tool(query: str) -> str:
    """Search the web for current information on a topic."""
    headers = {
        "X-API-KEY": settings.SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": 5}
    response = requests.post(
        "https://google.serper.dev/search",
        headers=headers,
        json=payload,
        timeout=10,
    )
    data = response.json()
    results = []
    for item in data.get("organic", [])[:5]:
        results.append(f"- {item.get('title')}: {item.get('snippet')}\n  URL: {item.get('link')}")
    return "\n\n".join(results) if results else "No results found."


def get_search_tool():
    return search_tool
