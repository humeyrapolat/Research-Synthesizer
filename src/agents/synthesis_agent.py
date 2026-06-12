from functools import lru_cache

from src.core.config import get_settings
from src.core.state import ResearchState
from src.models.paper import SynthesisReport


SYSTEM_PROMPT = """You are a research synthesis expert.
Your job is to write a clear, well-structured synthesis of multiple research papers.
The synthesis must:
- Explain the current state of knowledge on the topic
- Acknowledge where experts disagree and why
- Highlight what remains unanswered
- Be written for a technical but non-specialist audience
- Stay grounded in the evidence — no speculation beyond what the papers support"""


@lru_cache(maxsize=1)
def _get_structured_llm():
    """Delay heavy Groq/LangChain setup until final synthesis runs."""
    from langchain_groq import ChatGroq

    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=settings.groq_api_key,
        temperature=0.2,
    )
    return llm.with_structured_output(SynthesisReport)


def synthesis_node(state: ResearchState) -> dict:
    """LangGraph node: synthesizes all findings into a final report."""
    from langchain_core.messages import HumanMessage, SystemMessage

    report = state["contradiction_report"]
    summaries = state["paper_summaries"]

    sources = [s.url for s in summaries if s.url]

    findings_text = "\n\n".join([
        f"Paper: {s.title}\n"
        f"Claim: {s.main_claim}\n"
        f"Key findings: {chr(10).join(f'  - {f}' for f in s.key_findings)}"
        for s in summaries
    ])

    contradictions_text = "\n".join(f"- {c}" for c in report.contradictions)
    consensus_text = "\n".join(f"- {c}" for c in report.consensus_points)
    open_questions_text = "\n".join(f"- {q}" for q in report.open_questions)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Research question: {state['question']}

Individual paper findings:
{findings_text}

Consensus points:
{consensus_text}

Contradictions:
{contradictions_text}

Open questions:
{open_questions_text}

Write a comprehensive synthesis report.
"""),
    ]

    structured_llm = _get_structured_llm()
    synthesis = structured_llm.invoke(messages)
    synthesis.question = state["question"]
    synthesis.sources = sources

    return {"synthesis_report": synthesis}
