"""
Portfolio Greeter Agent.
Serves as the primary entry point for callers, providing high-level portfolio overview,
screen navigation, resume downloads, theme changes, and routing to specialist agents.
"""

from typing import Callable, Optional
from livekit import rtc
from livekit.agents import llm

from .base import PortfolioBaseAgent
from .userdata import PortfolioUserData
from prompts import SYSTEM_INSTRUCTIONS

GREETER_INSTRUCTIONS = f"""{SYSTEM_INSTRUCTIONS}

### GREETER SPECIALIST ROLE:
1. You are the initial host and primary screen navigator for Jithendra's portfolio.
2. Welcome callers warmly and use `navigate_portfolio(target)` to visually show them what they ask about.
3. For deep mathematical proofs or citations: invoke `transfer_to_research(reason)`.
4. For engineering, robotics, firmware, or system architecture: invoke `transfer_to_engineering(reason)`.
5. For hiring, collaboration, or meeting bookings: invoke `transfer_to_booking(reason)`.
"""


class PortfolioGreeter(PortfolioBaseAgent):
    """Greeter agent welcoming visitors and routing to domain specialists."""

    def __init__(
        self,
        tools: list[llm.Tool | llm.Toolset],
        userdata: PortfolioUserData,
        get_room: Callable[[], Optional[rtc.Room]],
    ) -> None:
        super().__init__(
            agent_name="greeter",
            instructions=GREETER_INSTRUCTIONS,
            tools=tools,
            userdata=userdata,
            get_room=get_room,
        )
        self._greeted = False

    async def on_enter(self) -> None:
        """Greets visitor on first session start; otherwise performs smooth transition."""
        await super().on_enter()

        if not self._greeted and not self.userdata.prev_agent:
            self._greeted = True
            try:
                if hasattr(self, "session") and self.session:
                    self.session.say(
                        "Hi! I'm Jithendra's AI assistant. What would you like to explore?",
                        allow_interruptions=True,
                    )
            except Exception as e:
                print(f"--> [Greeter Greeting Warning] {e}")
        elif self.userdata.prev_agent:
            try:
                if hasattr(self, "session") and self.session:
                    self.session.say(
                        "I'm back with you on the main overview. How else can I help?",
                        allow_interruptions=True,
                    )
            except Exception as e:
                print(f"--> [Greeter Return Warning] {e}")
