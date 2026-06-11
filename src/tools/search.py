from tavily import TavilyClient
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search_research_papers(query: str, max_results: int = 12) -> list[dict]:
    """Search for academic papers and research content on a given topic."""
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