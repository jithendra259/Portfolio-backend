import json
from typing import Annotated, Any, Callable

from livekit import rtc
from livekit.agents import llm


async def broadcast_navigation(room: rtc.Room | None, target: str) -> str:
    """
    Publishes a JSON navigation packet over the LiveKit data channel 'navigation'.
    Directs the frontend to scroll or switch routes to the requested target.
    """
    print(f"--> [Agent Tool] Navigating frontend to: {target}")

    if room and hasattr(room, "local_participant") and room.local_participant:
        try:
            payload = json.dumps({"type": "navigate", "target": target})
            await room.local_participant.publish_data(payload.encode("utf-8"), topic="navigation")
            print(f"--> [Agent Data Channel] Published navigation packet for '{target}' successfully.")
            return f"Successfully navigated screen to {target}."
        except Exception as e:
            print(f"--> [Agent Error] Failed to publish navigation packet: {e}")
            return f"Attempted navigation to {target}: {e}"

    print(f"--> [Agent Warning] No active room participant found to publish navigation.")
    return f"Navigation requested for {target}."


from prompts.knowledge.pages import (
    get_formatted_section_explanation,
    get_section_knowledge,
)


class NavigationToolset(llm.Toolset):
    """Modular toolset for real-time frontend screen navigation, subsection scrolling, and section awareness."""

    def __init__(
        self,
        get_room: Callable[[], rtc.Room | None],
        get_assistant: Callable[[], Any] | None = None,
    ) -> None:

        @llm.function_tool(
            description="Navigate visitor's screen in real time to any section ('contact', 'skills', 'projects', 'research', 'experience', 'about', 'home', 'book_appointment', or case study name)."
        )
        async def navigate_portfolio(
            target: Annotated[str, "Target section, page, or case study name."],
        ) -> str:
            """Navigates user screen to the desired section."""
            room = get_room()
            clean_target = target.strip().lower().replace("#", "")
            await broadcast_navigation(room, clean_target)
            explanation = get_formatted_section_explanation(clean_target)
            return f"Navigated screen to {clean_target}. Details: {explanation}"

        @llm.function_tool(
            description="Explain what a portfolio section or subsection contains."
        )
        async def explain_section(
            section_name: Annotated[str, "Name of the section (e.g. 'contact', 'skills', 'research')."],
        ) -> str:
            """Provides explanation of what a section tells."""
            return get_formatted_section_explanation(section_name)

        @llm.function_tool(
            description="Query what page or case study the visitor is currently viewing."
        )
        async def get_current_page_context() -> str:
            """Inspects and returns the visitor's current screen and page context concisely."""
            assistant = get_assistant() if get_assistant else None
            if assistant and hasattr(assistant, "get_formatted_page_context"):
                context_str = assistant.get_formatted_page_context()
                print(f"--> [Agent Tool] get_current_page_context: {context_str[:120]}...")
                return context_str
            return "The visitor is on the main portfolio page (/)."

        @llm.function_tool(
            description="List portfolio sections and case studies available to view."
        )
        async def list_portfolio_pages() -> str:
            """Returns directory of available sections."""
            return "Sections: home, about, research, projects, skills, experience, certificates, resume, contact. Booking: /book-appointment."

        super().__init__(
            id="navigation",
            tools=[navigate_portfolio, explain_section, get_current_page_context, list_portfolio_pages],
        )


class ResearchToolset(llm.Toolset):
    """Modular toolset for deep mathematical analysis and peer-reviewed publication breakdowns."""

    def __init__(
        self,
        get_session: Callable[[], Any],
        get_room: Callable[[], rtc.Room | None],
    ) -> None:

        from .reasoner import ResearchReasoner

        @llm.function_tool(
            description=(
                "Perform in-depth technical analysis and mathematical breakdowns for Jithendra's 3 research papers "
                "(Elsevier EAAI, Springer Nature LNCS, Elsevier COR) or engineering projects (AQI Forecasting, Swarm Robotics). "
                "Provides mathematical formulations (G-CVaR, Ledoit-Wolf alpha=0.42, CLARABEL interior-point SOCP solver), "
                "architecture graphs, and citation details while speaking natural fillers."
            )
        )
        async def research_paper_deep_dive(
            topic: Annotated[
                str,
                "The research paper title, mathematical concept (e.g. G-CVaR, Ledoit-Wolf, CLARABEL), or project name.",
            ],
        ) -> str:
            """Executes background research reasoner with non-blocking spoken filler."""
            session = get_session()
            room = get_room()
            return await ResearchReasoner.analyze_topic(session, room, topic)

        @llm.function_tool(
            description=(
                "Query and retrieve a precise factual metric or formulation from Jithendra's research papers "
                "or engineering specifications. Returns a concise single-fact answer (under 160 chars)."
            )
        )
        async def semantic_knowledge_search(
            query: Annotated[
                str,
                "The technical question or topic to retrieve a specific metric or formulation for.",
            ],
        ) -> str:
            """Executes vector retrieval returning only the single top fact."""
            from .rag import search_knowledge_base
            results = search_knowledge_base(query=query, top_k=1)
            if not results:
                return f"No direct record found for '{query}'."
            top_chunk = results[0]
            raw_text = (top_chunk.get("text") or top_chunk.get("content") or "").strip()
            clean_text = " ".join(raw_text.split())
            if clean_text.startswith("[PDF"):
                idx = clean_text.find("]")
                if idx != -1:
                    clean_text = clean_text[idx + 1:].strip()
            single_fact = clean_text[:160]
            return f"Fact: {single_fact}"

        super().__init__(id="research", tools=[research_paper_deep_dive, semantic_knowledge_search])


class ResourceToolset(llm.Toolset):
    """Voice-triggered downloads for public portfolio resources."""

    def __init__(self, get_room: Callable[[], rtc.Room | None]) -> None:
        @llm.function_tool(
            description=(
                "Download a public portfolio resource in the visitor's browser. "
                "Supported resources: resume, research, certificates, aqi_report, and swarm_report."
            )
        )
        async def download_resource(
            resource: Annotated[
                str,
                "One of resume, research, certificates, aqi_report, or swarm_report.",
            ],
        ) -> str:
            resources = {
                "resume": ("/documents/resume/kandula_jithendra_subramanyam_resume.pdf", "Kandula_Jithendra_Subramanyam_Resume.pdf"),
                "research": ("/documents/adaptive-portfolio-governance/multi-agent-governance-graph-cvar-eaai.pdf", "Jithendra_EAAI_Research.pdf"),
                "certificates": ("/certificates/conference/ijcaci-2026-paper-presentation.jpg", "Jithendra_IJCACI_Certificate.jpg"),
                "aqi_report": ("/documents/personalised-aqi-system/mtech-miniproject-aqi-forecasting-kandula-subramanyam.pdf", "Jithendra_AQI_Report.pdf"),
                "swarm_report": ("/documents/swarm-robots-agriculture/swarm-robotics-btech-report.docx", "Jithendra_Swarm_Robotics_Report.docx"),
            }
            key = resource.strip().lower().replace(" ", "_")
            if key not in resources:
                return "I can download the resume, research paper, certificates, AQI report, or swarm robotics report."

            room = get_room()
            if room and room.local_participant:
                url, filename = resources[key]
                payload = json.dumps({"type": "download", "url": url, "filename": filename})
                await room.local_participant.publish_data(payload.encode("utf-8"), topic="assistant_action")
                return f"Starting the {key.replace('_', ' ')} download now."
            return "The download is ready, but the browser action channel is not connected yet."

        super().__init__(id="resources", tools=[download_resource])


class ThemeToolset(llm.Toolset):
    """Voice control for the portfolio color theme."""

    def __init__(self, get_room: Callable[[], rtc.Room | None]) -> None:
        @llm.function_tool(
            description=(
                "Change the portfolio appearance when the visitor asks for dark mode, light mode, "
                "night mode, or day mode. The theme must be exactly dark or light."
            )
        )
        async def set_theme(
            theme: Annotated[str, "The requested theme: dark or light."],
        ) -> str:
            selected_theme = theme.strip().lower()
            if selected_theme not in {"dark", "light"}:
                return "I can switch the portfolio between dark and light mode."

            room = get_room()
            if room and room.local_participant:
                payload = json.dumps({"type": "theme", "theme": selected_theme})
                await room.local_participant.publish_data(
                    payload.encode("utf-8"),
                    topic="assistant_action",
                )
                return f"Switched the portfolio to {selected_theme} mode."
            return "The theme control is unavailable until the browser connection is ready."

        super().__init__(id="theme", tools=[set_theme])
