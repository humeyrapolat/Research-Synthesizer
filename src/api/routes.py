import logging
import uuid

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.core.graph import get_research_graph
from src.core.langfuse_config import get_langfuse_handler

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Synthesizer API",
    version="0.1.0",
    description="Multi-agent research synthesis API powered by LangGraph.",
)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class ResearchResponse(BaseModel):
    session_id: str
    question: str
    summary: str
    contradictions: list[str]
    open_questions: list[str]
    sources: list[str]


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    session_id = str(uuid.uuid4())
    langfuse_handler = get_langfuse_handler(session_id=session_id)
    callbacks = [langfuse_handler] if langfuse_handler else []
    research_graph = get_research_graph()

    try:
        result = await research_graph.ainvoke(
            input={
                "question": question,
                "search_results": [],
                "paper_summaries": [],
                "contradiction_report": None,
                "synthesis_report": None,
            },
            config={"callbacks": callbacks},
        )
    except RuntimeError as exc:
        logger.exception("Research pipeline configuration error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Research pipeline failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Research pipeline failed",
        ) from exc
    finally:
        if langfuse_handler:
            langfuse_handler.flush()

    report = result.get("synthesis_report")
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Research pipeline did not produce a synthesis report",
        )

    return ResearchResponse(
        session_id=session_id,
        question=report.question,
        summary=report.summary,
        contradictions=report.contradictions,
        open_questions=report.open_questions,
        sources=report.sources,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
