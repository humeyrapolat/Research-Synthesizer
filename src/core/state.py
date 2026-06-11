from typing import TypedDict, Annotated
import operator
from src.models.paper import PaperSummary, ContradictionReport, SynthesisReport

class ResearchState(TypedDict):
    question: str
    search_results: list[dict]     
    paper_summaries: Annotated[list[PaperSummary], operator.add] 
    contradiction_report: ContradictionReport | None
    synthesis_report: SynthesisReport | None