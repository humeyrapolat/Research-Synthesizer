from functools import lru_cache


def build_graph():
    from langgraph.graph import END, StateGraph

    from src.agents.contradiction_agent import contradiction_node
    from src.agents.reader_agent import reader_node
    from src.agents.search_agent import search_node
    from src.agents.synthesis_agent import synthesis_node
    from src.core.state import ResearchState

    graph = StateGraph(ResearchState)

    graph.add_node("search", search_node)
    graph.add_node("reader", reader_node)
    graph.add_node("contradiction", contradiction_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("search")
    graph.add_edge("search", "reader")
    graph.add_edge("reader", "contradiction")
    graph.add_edge("contradiction", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_research_graph():
    return build_graph()
