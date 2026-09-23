"""
Handoff Tools for Multi-Agent Portfolio System.
Enables seamless LiveKit agent-to-agent transfers using session.update_agent
with shared userdata preservation, matching the LiveKit Restaurant Agent recipe.
"""

from typing import Annotated, Any, Callable
from livekit.agents import llm
from .userdata import PortfolioUserData, HandoffPacket
from agent.supabase_logger import log_turn


def create_handoff_tools(userdata: PortfolioUserData, get_session: Callable[[], Any]) -> dict[str, llm.FunctionTool]:
    """Factory creating scoped handoff tools bound to the active session and shared userdata."""

    @llm.function_tool(
        description="Transfer to Research Specialist for math proofs, G-CVaR, Ledoit-Wolf, or 3 papers."
    )
    async def transfer_to_research(
        reason: Annotated[str, "The research topic or paper to explore."],
    ):
        """Transfers active agent to ResearchSpecialist."""
        session = get_session()
        target = userdata.agents.get("research")
        if not session or not target:
            return "Research Specialist is ready to answer your technical questions right now."

        current = getattr(session, "current_agent", None) or userdata.agents.get("greeter")
        userdata.prev_agent = current
        userdata.record_topic(f"Research: {reason}")
        last_query = getattr(current, "last_user_query", "") or reason

        userdata.pending_handoff = HandoffPacket(
            target="research",
            reason=reason,
            active_screen=userdata.active_screen,
            last_user_query=last_query,
        )

        log_turn(
            session_id=userdata.session_id,
            role="system",
            content=f"Handoff from {getattr(current, 'agent_name', 'agent')} to research: {reason}",
            active_agent="research",
            target_screen=userdata.active_screen,
        )

        session.update_agent(target)
        return target, f"Transferring you to our Research Specialist to examine {reason} in depth."

    @llm.function_tool(
        description="Transfer to Engineering Specialist for robotics, swarm ESP32 mesh, AQI, or voice AI."
    )
    async def transfer_to_engineering(
        reason: Annotated[str, "The project or architecture to discuss."],
    ):
        """Transfers active agent to EngineeringSpecialist."""
        session = get_session()
        target = userdata.agents.get("engineering")
        if not session or not target:
            return "Engineering Specialist is ready to discuss technical implementations right now."

        current = getattr(session, "current_agent", None) or userdata.agents.get("greeter")
        userdata.prev_agent = current
        userdata.record_topic(f"Engineering: {reason}")
        last_query = getattr(current, "last_user_query", "") or reason

        userdata.pending_handoff = HandoffPacket(
            target="engineering",
            reason=reason,
            active_screen=userdata.active_screen,
            last_user_query=last_query,
        )

        log_turn(
            session_id=userdata.session_id,
            role="system",
            content=f"Handoff from {getattr(current, 'agent_name', 'agent')} to engineering: {reason}",
            active_agent="engineering",
            target_screen=userdata.active_screen,
        )

        session.update_agent(target)
        return target, f"Connecting you with our Engineering Specialist to discuss {reason}."

    @llm.function_tool(
        description="Transfer to Booking Specialist to schedule a meeting, interview, or call."
    )
    async def transfer_to_booking(
        reason: Annotated[str, "The purpose of the meeting or interview."],
    ):
        """Transfers active agent to BookingSpecialist."""
        session = get_session()
        target = userdata.agents.get("booking")
        if not session or not target:
            return "Booking Specialist is ready to arrange a meeting right now."

        current = getattr(session, "current_agent", None) or userdata.agents.get("greeter")
        userdata.prev_agent = current
        userdata.record_topic(f"Booking: {reason}")
        last_query = getattr(current, "last_user_query", "") or reason

        userdata.pending_handoff = HandoffPacket(
            target="booking",
            reason=reason,
            active_screen="/book-appointment",
            last_user_query=last_query,
        )

        log_turn(
            session_id=userdata.session_id,
            role="system",
            content=f"Handoff from {getattr(current, 'agent_name', 'agent')} to booking: {reason}",
            active_agent="booking",
            target_screen="/book-appointment",
        )

        session.update_agent(target)
        return target, "Transferring you to our Booking Specialist to reserve time on Jithendra's calendar."

    @llm.function_tool(
        description="Transfer back to Greeter agent for general portfolio overview or navigation."
    )
    async def transfer_to_greeter(
        reason: Annotated[str, "Reason for returning to main overview."],
    ):
        """Transfers active agent back to PortfolioGreeter."""
        session = get_session()
        target = userdata.agents.get("greeter")
        if not session or not target:
            return "Welcome back to the main portfolio overview."

        current = getattr(session, "current_agent", None)
        userdata.prev_agent = current
        last_query = getattr(current, "last_user_query", "") or reason

        userdata.pending_handoff = HandoffPacket(
            target="greeter",
            reason=reason,
            active_screen=userdata.active_screen,
            last_user_query=last_query,
        )

        log_turn(
            session_id=userdata.session_id,
            role="system",
            content=f"Handoff back to greeter: {reason}",
            active_agent="greeter",
            target_screen=userdata.active_screen,
        )

        session.update_agent(target)
        return target, "Returning to the main portfolio overview."

    @llm.function_tool(
        description="Record or update visitor's name and email."
    )
    async def update_visitor_info(
        name: Annotated[str, "Visitor's full name."] = "",
        email: Annotated[str, "Visitor's email address."] = "",
    ) -> str:
        """Records visitor contact details in shared userdata."""
        userdata.set_visitor(name=name, email=email)
        return f"Visitor profile updated: name='{userdata.visitor_name or 'N/A'}', email='{userdata.visitor_email or 'N/A'}'."

    return {
        "research": transfer_to_research,
        "engineering": transfer_to_engineering,
        "booking": transfer_to_booking,
        "greeter": transfer_to_greeter,
        "visitor_info": update_visitor_info,
    }
