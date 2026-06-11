from langgraph.graph import StateGraph, END
from src.core.state import ResearchState
from src.agents.search_agent import search_node
from src.agents.reader_agent import reader_node
from src.agents.contradiction_agent import contradiction_node
from src.agents.synthesis_agent import synthesis_node


def build_graph():
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


research_graph = build_graph()