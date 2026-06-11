from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.models.paper import PaperSummary
from src.core.state import ResearchState
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

structured_llm = llm.with_structured_output(PaperSummary)

SYSTEM_PROMPT = """You are a research paper analyst.
Extract structured information from the given paper content.
Be precise and concise. If information is missing, make a reasonable inference.
Confidence score: 1.0 = very clear content, 0.5 = partial content, 0.2 = very sparse."""


async def summarize_single_paper(paper: dict) -> PaperSummary:
    """Summarizes a single paper asynchronously."""
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