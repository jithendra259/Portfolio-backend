"""
Multi-Agent Specialist Architecture for Portfolio Voice AI.
Implements the LiveKit Restaurant Agent pattern with typed session.userdata,
specialist handoffs, bounded context preservation, and non-blocking Supabase logging.
"""

from typing import Any, Callable, Optional, Tuple
from livekit import rtc

from agent.tools import (
    NavigationToolset,
    ResearchToolset,
    ResourceToolset,
    ThemeToolset,
)
from .userdata import PortfolioUserData
from .tools import create_handoff_tools
from .greeter import PortfolioGreeter
from .research import ResearchSpecialist
from .engineering import EngineeringSpecialist
from .booking import BookingSpecialist


def create_multi_agent_system(
    session_id: str,
    get_room: Callable[[], Optional[rtc.Room]],
    get_session: Callable[[], Any],
) -> Tuple[PortfolioGreeter, PortfolioUserData]:

    """
    Constructs the 4-agent specialist cluster and shared PortfolioUserData.
    Returns the primary entry agent (PortfolioGreeter) and the shared userdata.
    """
    userdata = PortfolioUserData(session_id=session_id)

    # 1. Base modular toolsets
    nav_toolset = NavigationToolset(get_room, get_assistant=lambda: userdata.agents.get("greeter"))
    research_toolset = ResearchToolset(get_session, get_room)
    resource_toolset = ResourceToolset(get_room)
    theme_toolset = ThemeToolset(get_room)

    # 2. Handoff tools bound to session and userdata
    handoff_tools = create_handoff_tools(userdata, get_session)

    # 3. Assemble agent tool allocations
    greeter_tools = [nav_toolset, resource_toolset, theme_toolset] + handoff_tools
    research_tools = [research_toolset, nav_toolset] + handoff_tools
    engineering_tools = [research_toolset, resource_toolset, nav_toolset] + handoff_tools
    booking_tools = [nav_toolset] + handoff_tools

    # 4. Instantiate specialist agents
    greeter = PortfolioGreeter(tools=greeter_tools, userdata=userdata, get_room=get_room)
    research = ResearchSpecialist(tools=research_tools, userdata=userdata, get_room=get_room)
    engineering = EngineeringSpecialist(tools=engineering_tools, userdata=userdata, get_room=get_room)
    booking = BookingSpecialist(tools=booking_tools, userdata=userdata, get_room=get_room)

    # 5. Register in shared state dictionary
    userdata.agents = {
        "greeter": greeter,
        "research": research,
        "engineering": engineering,
        "booking": booking,
    }

    return greeter, userdata


__all__ = [
    "PortfolioUserData",
    "PortfolioGreeter",
    "ResearchSpecialist",
    "EngineeringSpecialist",
    "BookingSpecialist",
    "create_multi_agent_system",
]
