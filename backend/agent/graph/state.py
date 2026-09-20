"""
Graph State definition for the Portfolio LangGraph workflow.
"""

from typing import Any, TypedDict


class PortfolioGraphState(TypedDict, total=False):
    """Execution state container passed through LangGraph nodes."""
    query: str
    route: str
    intent: str
    target: str
    screen_context: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    context: str
    grounding: str
    confidence: float
    user_intent: str
    user_expectation: str
    description_mode: str  # 'short' (default < 20 words) or 'long' (deep dive 45-75 words)

