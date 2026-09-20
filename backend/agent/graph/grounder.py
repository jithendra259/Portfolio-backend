"""
Screen grounder node for the LangGraph workflow.
Injects real-time WebRTC telemetry and structured page context.
"""

from typing import Any
from prompts.knowledge import PAGE_KNOWLEDGE
from .state import PortfolioGraphState


def screen_grounder_node(state: PortfolioGraphState) -> dict[str, Any]:
    """Enriches state with structured knowledge about what the visitor is actively viewing on their screen."""
    screen_ctx = state.get("screen_context", {})
    pathname = (screen_ctx.get("pathname") or "/").strip()

    data = PAGE_KNOWLEDGE.get(pathname)
    if not data:
        for k, v in PAGE_KNOWLEDGE.items():
            if k != "/" and k in pathname:
                data = v
                break

    if data:
        title = data.get("title", pathname)
        summary = data.get("summary", "").strip()
        if len(summary) > 140:
            summary = summary[:137] + "..."
        screen_summary = f"Viewing '{title}' ({pathname}): {summary}"
    else:
        screen_summary = f"Viewing page '{pathname}'."

    return {"context": screen_summary}
