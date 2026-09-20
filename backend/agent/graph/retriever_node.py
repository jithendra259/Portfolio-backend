"""
Vector retrieval node for the LangGraph workflow.
Integrates with the modular RAG engine for semantic search over publications and projects.
"""

from typing import Any
from agent.rag import search_knowledge_base
from .state import PortfolioGraphState


def vector_retrieval_node(state: PortfolioGraphState) -> dict[str, Any]:
    """Retrieves top semantic vector chunks from research papers, case studies, and candidate profile."""
    query = state.get("query", "")
    route = state.get("route", "")

    # Category filter if applicable
    category = None
    if route.startswith("research_"):
        category = route

    try:
        chunks = search_knowledge_base(query=query, top_k=2, category=category)
    except Exception as e:
        print(f"--> [LangGraph Retrieval Warning] Vector search error: {e}")
        chunks = []

    return {"retrieved_chunks": chunks}
