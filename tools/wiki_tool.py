from crewai.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper

_wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)

@tool("Wikipedia Search")
def wiki_tool(query: str) -> str:
    """Search Wikipedia for factual information about a topic."""
    return _wiki.run(query)


def get_wiki_tool():
    return wiki_tool
