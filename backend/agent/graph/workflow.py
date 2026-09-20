"""
LangGraph compilation and execution workflow.
Builds and compiles the StateGraph, defines conditional routing edges, and provides asynchronous entrypoint.
"""

from typing import Any
from langgraph.graph import END, START, StateGraph

from .classifier import classify_intent_node
from .grounder import screen_grounder_node
from .retriever_node import vector_retrieval_node
from .synthesizer import ground_and_synthesize_node
from .state import PortfolioGraphState


def _route_decision(state: PortfolioGraphState) -> str:
    """Conditional edge router determining the optimal sub-graph execution path."""
    route = state.get("route", "")
    if route in ("resource", "theme", "booking", "navigation"):
        # Fast path: skip vector retrieval to maintain sub-3ms latency for direct commands
        return "synthesize"
    elif route == "current_page":
        # Screen awareness path: inspect screen first, then augment with vector retrieval
        return "screen_grounder"
    else:
        # Technical, project, or general inquiry: run vector retrieval
        return "vector_retrieval"


def _build_portfolio_graph():
    """Compiles the LangGraph state machine."""
    builder = StateGraph(PortfolioGraphState)

    # 1. Add functional processing nodes
    builder.add_node("classify", classify_intent_node)
    builder.add_node("screen_grounder", screen_grounder_node)
    builder.add_node("vector_retrieval", vector_retrieval_node)
    builder.add_node("synthesize", ground_and_synthesize_node)

    # 2. Add edges
    builder.add_edge(START, "classify")

    # Conditional branch from classify
    builder.add_conditional_edges(
        "classify",
        _route_decision,
        {
            "synthesize": "synthesize",
            "screen_grounder": "screen_grounder",
            "vector_retrieval": "vector_retrieval",
        },
    )

    # Screen grounder proceeds directly to synthesis for active page awareness
    builder.add_edge("screen_grounder", "synthesize")

    # Vector retrieval proceeds to synthesis
    builder.add_edge("vector_retrieval", "synthesize")

    # Final grounding outputs to END
    builder.add_edge("synthesize", END)

    return builder.compile()


# Global compiled LangGraph instance
PORTFOLIO_GRAPH = _build_portfolio_graph()


async def route_portfolio_query(
    query: str,
    screen_context: dict[str, Any] | None = None,
) -> PortfolioGraphState:
    """
    Asynchronously executes the LangGraph workflow for a visitor utterance.
    Integrates query classification, screen awareness, and hybrid vector RAG.
    """
    initial_state: PortfolioGraphState = {
        "query": query,
        "screen_context": screen_context or {},
    }
    return await PORTFOLIO_GRAPH.ainvoke(initial_state)
