import asyncio
from functools import lru_cache

from src.core.config import get_settings
from src.core.state import ResearchState
from src.models.paper import PaperSummary


@lru_cache(maxsize=1)
def _get_structured_llm():
    """Delay heavy Groq/LangChain setup until summaries are requested."""
    from langchain_groq import ChatGroq

    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=settings.groq_api_key,
        temperature=0,
    )
    return llm.with_structured_output(PaperSummary)


SYSTEM_PROMPT = """You are a research paper analyst.
Extract structured information from the given paper content.
Be precise and concise. If information is missing, make a reasonable inference.
Confidence score: 1.0 = very clear content, 0.5 = partial content, 0.2 = very sparse."""


async def summarize_single_paper(paper: dict) -> PaperSummary:
    """Summarizes a single paper asynchronously."""
    from langchain_core.messages import HumanMessage, SystemMessage

    structured_llm = _get_structured_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Title: {paper['title']}
URL: {paper['url']}
Content: {paper['content'][:3000]}

Extract structured information from this paper.
"""),
    ]
    return await structured_llm.ainvoke(messages)


async def reader_node(state: ResearchState) -> dict:
    """LangGraph node: summarizes all papers in parallel."""
    papers = state["search_results"]

    tasks = [summarize_single_paper(paper) for paper in papers]
    summaries = await asyncio.gather(*tasks, return_exceptions=True)

    valid_summaries = [
        s for s in summaries
        if isinstance(s, PaperSummary)
    ]

    return {"paper_summaries": valid_summaries}
