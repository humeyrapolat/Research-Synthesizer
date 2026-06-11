from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.models.paper import ContradictionReport
from src.core.state import ResearchState
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

structured_llm = llm.with_structured_output(ContradictionReport)

SYSTEM_PROMPT = """You are a critical research analyst.
Your job is to compare multiple paper summaries and identify:
1. Direct contradictions — where papers claim opposite things
2. Consensus points — where all or most papers agree
3. Open questions — important gaps no paper has answered

Be specific. Reference findings by content, not by paper number.
If evidence is one-sided, still report it honestly."""


def contradiction_node(state: ResearchState) -> dict:
    """LangGraph node: detects contradictions across paper summaries."""
    summaries = state["paper_summaries"]

    if not summaries:
        return {
            "contradiction_report": ContradictionReport(
                contradictions=[],
                consensus_points=[],
                open_questions=["No papers found to analyze."],
            )
        }

    summaries_text = "\n\n".join([
        f"Paper: {s.title}\n"
        f"Claim: {s.main_claim}\n"
        f"Method: {s.method}\n"
        f"Findings: {chr(10).join(f'  - {f}' for f in s.key_findings)}"
        for s in summaries
    ])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Research question: {state['question']}

Paper summaries:
{summaries_text}

Analyze these papers and identify contradictions, consensus points, and open questions.
"""),
    ]

    report = structured_llm.invoke(messages)
    return {"contradiction_report": report}