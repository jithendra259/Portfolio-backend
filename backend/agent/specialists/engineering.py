"""
Engineering Specialist Agent.
Specializes in real-world ML systems, Autonomous Swarm Robotics (ESP32 mesh),
AQI Forecasting (XGBoost), and the low-latency LiveKit Voice AI architecture.
"""

from typing import Callable, Optional
from livekit import rtc
from livekit.agents import llm

from .base import PortfolioBaseAgent
from .userdata import PortfolioUserData
from prompts import SYSTEM_INSTRUCTIONS, get_specialist_instructions

ENGINEERING_ROLE_INSTRUCTIONS = """
### ENGINEERING SPECIALIST ROLE:
1. Expert in: Voice AI Architecture (LiveKit, Groq LPU, Cartesia, Deepgram), Swarm Robotics (ESP32, ESP-NOW), AQI Forecasting (XGBoost), Agentic Systems (LangGraph).
2. Explain architectures, dataflows, and implementation details with technical precision.
3. Use `navigate_portfolio` to show engineering case studies.
4. Transfer: `transfer_to_research` (math), `transfer_to_greeter` (overview), `transfer_to_booking` (collab).
5. Short mode: <20 words. Long mode: 45-70 words, key result first.
"""


class EngineeringSpecialist(PortfolioBaseAgent):
    """Specialist agent focused on distributed systems, robotics, and production ML pipelines."""

    def __init__(
        self,
        tools: list[llm.Tool | llm.Toolset],
        userdata: PortfolioUserData,
        get_room: Callable[[], Optional[rtc.Room]],
    ) -> None:
        instructions = SYSTEM_INSTRUCTIONS + ENGINEERING_ROLE_INSTRUCTIONS
        super().__init__(
            agent_name="engineering",
            instructions=instructions,
            tools=tools,
            userdata=userdata,
            get_room=get_room,
        )

    async def on_enter(self) -> None:
        await super().on_enter()
        try:
            if hasattr(self, "session") and self.session:
                self.session.say(
                    "Engineering Specialist ready. We can inspect the ESP32 swarm mesh, XGBoost AQI pipeline, or this WebRTC voice architecture. What interests you?",
                    allow_interruptions=True,
                )
        except Exception as e:
            print(f"--> [Engineering Entry Warning] {e}")