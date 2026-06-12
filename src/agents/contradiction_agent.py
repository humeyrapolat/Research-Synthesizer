from functools import lru_cache

from src.core.config import get_settings
from src.core.state import ResearchState
from src.models.paper import ContradictionReport


@lru_cache(maxsize=1)
def _get_structured_llm():
    """Delay heavy Groq/LangChain setup until contradiction analysis runs."""
    from langchain_groq import ChatGroq

    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=settings.groq_api_key,
        temperature=0,
    )
    return llm.with_structured_output(ContradictionReport)


SYSTEM_PROMPT = """You are a critical research analyst.
Your job is to compare multiple paper summaries and identify:
1. Direct contradictions — where papers claim opposite things
2. Consensus points — where all or most papers agree
3. Open questions — important gaps no paper has answered

Be specific. Reference findings by content, not by paper number.
If evidence is one-sided, still report it honestly."""


def contradiction_node(state: ResearchState) -> dict:
    """LangGraph node: detects contradictions across paper summaries."""
    from langchain_core.messages import HumanMessage, SystemMessage

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

    structured_llm = _get_structured_llm()
    report = structured_llm.invoke(messages)
    return {"contradiction_report": report}
