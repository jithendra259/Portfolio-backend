"""
System Instructions Builder for Jithendra's Voice AI Assistant.
Formats biography, research papers, projects, and voice policies into an ultra-low-latency prompt.
"""

from .knowledge import (
    BIOGRAPHY,
    EDUCATION,
    COMPETITIVE_EXAMS,
    PUBLICATIONS,
    PROJECTS,
    TECHNICAL_SKILLS,
    WORK_EXPERIENCE,
    NAVIGATION_TARGETS,
)
from .response_policy import COMMON_RESPONSE_POLICY


def _format_publications() -> str:
    lines = []
    for i, pub in enumerate(PUBLICATIONS, 1):
        lines.append(f"- Paper {i} ({pub['venue']}): \"{pub['title']}\" (target: '{pub['target_nav']}')")
    return "\n".join(lines)


def _format_projects() -> str:
    lines = []
    for proj in PROJECTS:
        desc = proj.get("description", "")
        if len(desc) > 90:
            desc = desc[:87] + "..."
        lines.append(f"- {proj['name']} (target: '{proj['target_nav']}'): {desc}")
    return "\n".join(lines)


def _format_targets() -> str:
    return ", ".join([f"'{k}'" for k in NAVIGATION_TARGETS.keys()])


from livekit.agents.beta import Instructions


def build_system_instructions() -> str:
    """Ultra-compact system prompt under 800 chars to minimize TTFT and prevent token rate limits."""
    return (
        "You are the Voice AI Assistant for Kandula Jithendra Subramanyam "
        "(M.Tech AI Somaiya CGPA 8.06, B.Tech ECE Presidency CGPA 7.77, GATE 2024 DA). "
        "AI Systems Engineer & Quant Researcher with 3 papers in Elsevier EAAI, Springer Nature LNCS, and Elsevier COR. "
        "Email: kandulajithendrasubramanyam@gmail.com.\n"
        "RULES:\n"
        "1. Short Description Mode (Default): For standard questions, provide a crisp, natural answer strictly under 15-20 words.\n"
        "2. Long Description Mode: When asked to 'explain in detail', 'tell me more', or for a 'deep dive', give a rich technical breakdown in 45-70 words.\n"
        "3. Directly address the visitor's intent without preamble. Never speak markdown symbols (no asterisks or bullets).\n"
        "4. When visitor asks to see any section or project, call navigate_portfolio(target) immediately.\n"
        "5. Targets: 'contact', 'skills', 'projects', 'research', 'experience', 'about', 'home', 'book_appointment'.\n"
        "6. Transfer to specialists: transfer_to_research (math/theorems), transfer_to_engineering (robotics/code), transfer_to_booking (meetings).\n"
        f"{COMMON_RESPONSE_POLICY}"
    )



SYSTEM_INSTRUCTIONS = build_system_instructions()

