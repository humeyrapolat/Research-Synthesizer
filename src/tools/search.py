from langchain_core.tools import tool

from src.core.config import get_settings


def _get_tavily_client():
    """Create the Tavily client only when the search tool is actually called."""
    from tavily import TavilyClient

    settings = get_settings()
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")

    return TavilyClient(api_key=settings.tavily_api_key)


@tool
def search_research_papers(query: str, max_results: int = 12) -> list[dict]:
    """Search for academic papers and research content on a given topic."""
    client = _get_tavily_client()
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_domains=[
            "arxiv.org",
            "semanticscholar.org",
            "scholar.google.com",
            "researchgate.net",
            "acm.org",
            "ieee.org",
        ],
    )
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        }
        for r in response.get("results", [])
    ]
