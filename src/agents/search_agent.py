from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.tools.search import search_research_papers
from src.core.state import ResearchState
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

llm_with_tools = llm.bind_tools([search_research_papers])

SYSTEM_PROMPT = """You are a research search specialist.
Your job is to find relevant academic papers and research content for a given question.
Always use the search tool with a well-formed academic query.
Expand the query if needed to capture related concepts."""


def search_node(state: ResearchState) -> dict:
    """LangGraph node: searches for papers based on the research question."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Find academic papers about: {state['question']}"),
    ]

    response = llm_with_tools.invoke(messages)

    results = []
    for tool_call in response.tool_calls:
        if tool_call["name"] == "search_research_papers":
            results = search_research_papers.invoke(tool_call["args"])
            break

    return {"search_results": results}