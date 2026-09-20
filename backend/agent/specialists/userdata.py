"""
Shared State Data Model for LiveKit Multi-Agent Handoffs.
Based on the LiveKit Restaurant Agent pattern using typed session.userdata.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class HandoffPacket:
    """Laser-targeted task packet passed between agents during handoffs.
    Contains only what the next agent needs to execute the specific task."""
    target: str
    reason: str
    active_screen: str = "/"
    last_user_query: str = ""


@dataclass
class PortfolioUserData:
    """
    Session-wide shared state tracked across all specialist agents:
    - Greeter Agent (welcome, high-level bio, theme, downloads)
    - Research Specialist (EAAI, LNCS, COR, G-CVaR, CLARABEL, Ledoit-Wolf)
    - Engineering Specialist (Swarm Robotics, AQI Forecasting, Voice AI)
    - Booking Specialist (Recruiter interviews, calendar scheduling, lead capture)
    """

    session_id: str
    visitor_name: Optional[str] = None
    visitor_email: Optional[str] = None
    active_screen: str = "/"
    active_title: str = "Kandula Jithendra Subramanyam | AI & Quant Portfolio"
    screen_context: dict = field(default_factory=dict)
    explored_topics: list[str] = field(default_factory=list)
    booking_details: dict = field(default_factory=dict)
    agents: dict[str, Any] = field(default_factory=dict)
    prev_agent: Optional[Any] = None
    pending_handoff: Optional[HandoffPacket] = None

    def record_topic(self, topic: str) -> None:
        """Adds a topic to explored history if not already present."""
        cleaned = topic.strip()
        if cleaned and cleaned not in self.explored_topics:
            self.explored_topics.append(cleaned)

    def set_visitor(self, name: Optional[str] = None, email: Optional[str] = None) -> None:
        """Updates visitor identity info."""
        if name and name.strip():
            self.visitor_name = name.strip()
        if email and email.strip():
            self.visitor_email = email.strip()

    def get_summary(self) -> str:
        """
        Returns a concise YAML-style state summary injected into agent context on handoffs.
        Keeps token overhead <120 tokens for minimal TTFT.
        """
        lines = [
            "visitor_state:",
            f"  name: {self.visitor_name or 'Anonymous Visitor'}",
            f"  email: {self.visitor_email or 'Not provided'}",
            f"  current_screen: {self.active_screen}",
        ]
        if self.explored_topics:
            lines.append(f"  explored_topics: {', '.join(self.explored_topics[-4:])}")
        if self.booking_details:
            date = self.booking_details.get("date", "pending")
            time = self.booking_details.get("time", "pending")
            lines.append(f"  booking_status: date={date}, time={time}")
        return "\n".join(lines)
