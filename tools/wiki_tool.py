from crewai.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper

from guardrails.tool_output_guardrail import sanitize_untrusted_text, wrap_untrusted_block

_wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)

@tool("Wikipedia Search")
def wiki_tool(query: str) -> str:
    """Search Wikipedia for factual information about a topic."""
    raw = _wiki.run(query)
    return wrap_untrusted_block(sanitize_untrusted_text(raw, "wikipedia"), "wikipedia content")


def get_wiki_tool():
    return wiki_tool
