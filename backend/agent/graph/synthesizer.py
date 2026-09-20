"""
Synthesis node for the LangGraph workflow.
Compiles screen context, destination routes, and vector RAG citations into factual speech grounding,
dynamically choosing between Short Description Mode (< 20 words) and Long Description Mode (45-70 words).
"""

from typing import Any
from agent.cache import KNOWLEDGE_CACHE
from .state import PortfolioGraphState


def ground_and_synthesize_node(state: PortfolioGraphState) -> dict[str, Any]:
    """Synthesizes targeted speech grounding with context-aware intent and expectation thinking."""
    query = state.get("query", "").strip()
    route = state.get("route", "general_inquiry")
    screen_context = state.get("context", "").strip()
    retrieved_chunks = state.get("retrieved_chunks", [])
    user_intent = state.get("user_intent", "").strip()
    user_expectation = state.get("user_expectation", "").strip()
    mode = state.get("description_mode", "short")

    thinking_part = ""
    if user_intent and user_expectation:
        thinking_part = f"[Thinking: Intent: {user_intent} | Expects: {user_expectation}]"

    # 1. Check in-memory Knowledge Cache for pre-compiled high-performance descriptions (0.01ms hit)
    cached_fact = KNOWLEDGE_CACHE.get_grounding(route, mode)
    if cached_fact:
        tag = "Brief Fact" if mode == "short" else "Detailed Technical Breakdown"
        grounding = f"{thinking_part} [{tag}: {cached_fact}]".strip()
        result = {
            "grounding": grounding,
            "context": cached_fact,
            "user_intent": user_intent,
            "user_expectation": user_expectation,
            "description_mode": mode,
            "route": route,
        }
        if query:
            KNOWLEDGE_CACHE.put(query, mode, result)
        return result

    # 2. Action, conversational, and general routes: provide thinking guidance without prompt bloat
    if route in ("navigation", "theme", "resource", "booking", "greeting"):
        result = {
            "grounding": thinking_part,
            "context": screen_context,
            "user_intent": user_intent,
            "user_expectation": user_expectation,
            "description_mode": mode,
            "route": route,
        }
        if query:
            KNOWLEDGE_CACHE.put(query, mode, result)
        return result

    # 3. Screen awareness: active screen synopsis with thinking guidance
    if route == "current_page":
        screen_part = f"[Current Screen: {screen_context}]" if screen_context else ""
        grounding = f"{thinking_part} {screen_part}".strip()
        result = {
            "grounding": grounding,
            "context": screen_context,
            "user_intent": user_intent,
            "user_expectation": user_expectation,
            "description_mode": mode,
            "route": route,
        }
        if query:
            KNOWLEDGE_CACHE.put(query, mode, result)
        return result

    # 4. Technical and paper routes from vector retrieval
    if retrieved_chunks:
        if mode == "long":
            facts = []
            for c in retrieved_chunks[:2]:
                txt = " ".join((c.get("text") or c.get("content") or "").split())
                if "]:" in txt:
                    txt = txt.split("]:", 1)[1].strip()
                elif "]" in txt and txt.startswith("["):
                    txt = txt.split("]", 1)[1].strip()
                facts.append(txt[:180])
            combined_fact = " ".join(facts)[:320]
            grounding = f"{thinking_part} [Detailed Relevant Facts: {combined_fact}]".strip()
            context_snippet = combined_fact
        else:
            top_chunk = retrieved_chunks[0]
            raw_text = (top_chunk.get("text") or top_chunk.get("content") or "").strip()
            clean_text = " ".join(raw_text.split())
            if "]:" in clean_text:
                clean_text = clean_text.split("]:", 1)[1].strip()
            elif "]" in clean_text and clean_text.startswith("["):
                clean_text = clean_text.split("]", 1)[1].strip()
            single_fact = clean_text[:120]
            grounding = f"{thinking_part} [Brief Fact: {single_fact}]".strip()
            context_snippet = single_fact

        result = {
            "grounding": grounding,
            "context": context_snippet,
            "user_intent": user_intent,
            "user_expectation": user_expectation,
            "description_mode": mode,
            "route": route,
        }
        if query:
            KNOWLEDGE_CACHE.put(query, mode, result)
        return result

    result = {
        "grounding": thinking_part,
        "context": screen_context,
        "user_intent": user_intent,
        "user_expectation": user_expectation,
        "description_mode": mode,
        "route": route,
    }
    if query:
        KNOWLEDGE_CACHE.put(query, mode, result)
    return result
