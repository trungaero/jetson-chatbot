"""
Tool definitions for the LangGraph ReAct agent.
Add new tools here — they are automatically picked up by agent.py.
"""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


# ── Internet Search ──────────────────────────────────────────────────────────

_search = DuckDuckGoSearchRun()


@tool
def search_internet(query: str) -> str:
    """Search the internet for current information.
    Use this for: news, weather, sports scores, recent events, live prices,
    or any question that requires up-to-date data not in training knowledge.

    Args:
        query: A concise search query string.

    Returns:
        A text summary of the top search results.
    """
    try:
        result = _search.run(query)
        return result if result else "No results found for that query."
    except Exception as e:
        return f"Search failed: {e}"


# ── Tool Registry ────────────────────────────────────────────────────────────

def get_tools() -> list:
    """Return all available tools for the agent."""
    return [
        search_internet,
    ]
