import asyncio
from src.core.graph import research_graph


async def main():
    result = await research_graph.ainvoke({
        "question": "How to reduce hallucination in RAG systems?",
        "search_results": [],
        "paper_summaries": [],
        "contradiction_report": None,
        "synthesis_report": None,
    })

    report = result["synthesis_report"]

    print(f"\n{'='*60}")
    print(f"RESEARCH SYNTHESIS REPORT")
    print(f"{'='*60}")
    print(f"\nQuestion: {report.question}\n")
    print(f"Summary:\n{report.summary}\n")

    print("Contradictions:")
    for c in report.contradictions:
        print(f"  - {c}")

    print("\nOpen Questions:")
    for q in report.open_questions:
        print(f"  ? {q}")

    print(f"\nSources ({len(report.sources)}):")
    for s in report.sources:
        print(f"  {s}")


if __name__ == "__main__":
    asyncio.run(main())