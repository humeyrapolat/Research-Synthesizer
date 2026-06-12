from functools import lru_cache

from src.core.config import get_settings
from src.core.state import ResearchState


SYSTEM_PROMPT = """You are a research search specialist.
Your job is to find relevant academic papers and research content for a given question.
Always use the search tool with a well-formed academic query.
Expand the query if needed to capture related concepts."""


@lru_cache(maxsize=1)
def _get_search_dependencies():
    """Delay heavy LangChain/Groq/Tavily imports until the search node runs."""
    from langchain_groq import ChatGroq
    from src.tools.search import search_research_papers

    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=settings.groq_api_key,
        temperature=0,
    )
    return llm.bind_tools([search_research_papers]), search_research_papers


def search_node(state: ResearchState) -> dict:
    """LangGraph node: searches for papers based on the research question."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm_with_tools, search_tool = _get_search_dependencies()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Find academic papers about: {state['question']}"),
    ]

    response = llm_with_tools.invoke(messages)

    results = []
    for tool_call in getattr(response, "tool_calls", []):
        if tool_call["name"] == "search_research_papers":
            results = search_tool.invoke(tool_call["args"])
            break

    if not results:
        results = search_tool.invoke({"query": state["question"], "max_results": 12})

    return {"search_results": results}
