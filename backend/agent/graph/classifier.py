"""
Intent classifier node for the LangGraph workflow.
Performs token-level parsing and word boundary matching to route visitor queries with 0ms latency.
"""

import re
from typing import Any
from .routes import NAVIGATION_TARGETS, ROUTE_RULES
from .state import PortfolioGraphState


def classify_intent_node(state: PortfolioGraphState) -> dict[str, Any]:
    """Classifies user intent, expected delivery, and destination screen target with 0ms latency."""
    query = state.get("query", "").strip()
    query_lower = query.lower()
    query_tokens = set(re.findall(r"[a-z0-9_\-]+", query_lower))

    from agent.cache import KNOWLEDGE_CACHE, detect_description_mode
    description_mode = detect_description_mode(query)
    exp_suffix = " (Long Mode: detailed breakdown in 45-70 words)" if description_mode == "long" else " (Short Mode: strictly under 15-20 words)"

    # 0. Check high-performance LRU query cache (sub-0.05ms hit)
    cached = KNOWLEDGE_CACHE.get(query, description_mode)
    if cached:
        return cached

    # 1. Explicit navigation requests (e.g. "navigate to projects", "go to about")
    if any(nav_kw in query_lower for nav_kw in ("navigate", "go to", "take me to", "jump to", "scroll to", "open")):
        for target, description in NAVIGATION_TARGETS.items():
            clean_target = target.replace("_", " ")
            if clean_target in query_lower or any(token == target for token in query_tokens):
                return {
                    "route": "navigation",
                    "intent": "navigate",
                    "target": target,
                    "context": description,
                    "description_mode": description_mode,
                    "user_intent": f"Wants to navigate to '{clean_target}' section",
                    "user_expectation": f"Expects immediate screen scroll to '{clean_target}' and a crisp 1-sentence confirmation",
                }


    # 2. Resource downloads (resume, papers, certs)
    if any(w in query_tokens for w in ("download", "pdf", "cv", "resume")) or "offline copy" in query_lower:
        resource = "resume"
        if any(w in query_tokens for w in ("paper", "research", "publication", "manuscript", "eaai")):
            resource = "research"
        elif any(w in query_tokens for w in ("certificate", "credential", "gate", "award")):
            resource = "certificates"
        elif "aqi" in query_tokens:
            resource = "aqi_report"
        elif "swarm" in query_tokens:
            resource = "swarm_report"
        return {
            "route": "resource",
            "intent": "download",
            "target": resource,
            "description_mode": description_mode,
            "user_intent": f"Wants to download {resource} offline document",
            "user_expectation": f"Expects download_resource tool trigger and brief download status{exp_suffix}",
        }

    # 3. Scheduling & Meeting Booking
    if any(w in query_tokens for w in ("book", "schedule", "appointment", "meeting", "interview")) or "call jithendra" in query_lower:
        return {
            "route": "booking",
            "intent": "book_appointment",
            "target": "book_appointment",
            "description_mode": description_mode,
            "user_intent": "Wants to book a meeting or schedule an interview with Jithendra",
            "user_expectation": f"Expects guidance to the booking interface and prompt for meeting details{exp_suffix}",
        }

    # 4. Theme toggle
    if any(phrase in query_lower for phrase in ("dark mode", "light mode", "change theme", "night mode", "day mode")):
        theme = "dark" if "dark" in query_tokens or "night" in query_tokens else "light"
        return {
            "route": "theme",
            "intent": "set_theme",
            "target": theme,
            "description_mode": description_mode,
            "user_intent": f"Wants to switch website theme to {theme} mode",
            "user_expectation": f"Expects set_theme tool execution and brief confirmation",
        }

    # 5. Screen / Current Page Awareness
    if any(phrase in query_lower for phrase in (
        "where am i", "what page", "which page", "this page", "current page",
        "this screen", "what am i looking at", "explain this project",
        "tell me about this", "what is on my screen", "explain this paper"
    )):
        return {
            "route": "current_page",
            "intent": "inspect_screen",
            "description_mode": description_mode,
            "user_intent": "Inquiring about what is actively displayed on their screen",
            "user_expectation": f"Expects a breakdown explaining the active screen context{exp_suffix}",
        }

    # 6. Specific Research Papers & Projects
    for route, target, keywords in ROUTE_RULES:
        if any(re.search(rf"\b{re.escape(keyword)}\b", query_lower) for keyword in keywords):
            return {
                "route": route,
                "intent": "explore_project",
                "target": target,
                "description_mode": description_mode,
                "user_intent": f"Probing technical methodology or empirical results for {target}",
                "user_expectation": f"Expects core mathematical formulation, architectural design, or validated metric{exp_suffix}",
            }

    # 7. Candidate Bio / Experience / Skills / Education
    if any(phrase in query_lower for phrase in ("who are you", "tell me about yourself", "who is jithendra", "background", "biography")):
        return {
            "route": "profile",
            "intent": "answer_profile",
            "description_mode": description_mode,
            "user_intent": "Learning who Jithendra is and his engineering and research domain",
            "user_expectation": f"Expects high-level summary of AI engineering and research background{exp_suffix}",
        }

    if any(w in query_tokens for w in ("education", "degree", "college", "somaiya", "presidency", "cgpa", "gate")):
        return {
            "route": "education",
            "intent": "answer_education",
            "description_mode": description_mode,
            "user_intent": "Vetting academic qualifications and degrees",
            "user_expectation": f"Expects verified degree titles, universities, and CGPAs{exp_suffix}",
        }

    if any(w in query_tokens for w in ("skill", "skills", "stack", "technologies", "tools", "python", "pytorch", "cvxpy")):
        return {
            "route": "skills",
            "intent": "answer_skills",
            "description_mode": description_mode,
            "user_intent": "Evaluating engineering competencies and tech stack",
            "user_expectation": f"Expects primary production tools (Python, PyTorch, LangGraph, CVXPY/CLARABEL){exp_suffix}",
        }

    if any(w in query_tokens for w in ("experience", "work", "job", "appfabs", "role")):
        return {
            "route": "experience",
            "intent": "answer_experience",
            "description_mode": description_mode,
            "user_intent": "Reviewing Jithendra's industry experience as Applied AI Engineer at Appfabs",
            "user_expectation": f"Expects concise overview of industry work and engineering responsibilities{exp_suffix}",
        }

    # 8. Target matching without explicit "navigate" verb
    for target, description in NAVIGATION_TARGETS.items():
        clean_target = target.replace("_", " ")
        if clean_target in query_lower:
            return {
                "route": "navigation",
                "intent": "navigate",
                "target": target,
                "context": description,
                "description_mode": description_mode,
                "user_intent": f"Referencing portfolio section '{clean_target}'",
                "user_expectation": f"Expects screen navigation to '{clean_target}' and a relevant spoken highlight{exp_suffix}",
            }

    is_greeting = any(w in query_tokens for w in ("hi", "hello", "hey", "morning", "afternoon", "evening", "greetings"))
    user_intent = "Conversational greeting or general engagement" if is_greeting else "General portfolio question or discussion"
    user_expectation = (
        "Expects a polite, welcoming response strictly under 15-20 words offering assistance"
        if is_greeting
        else f"Expects concise guidance to portfolio sections, research papers, or booking{exp_suffix}"
    )
    return {
        "route": "general_inquiry",
        "intent": "semantic_answer",
        "description_mode": description_mode,
        "user_intent": user_intent,
        "user_expectation": user_expectation,
    }

