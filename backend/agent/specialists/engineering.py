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
from prompts.response_policy import COMMON_RESPONSE_POLICY

ENGINEERING_INSTRUCTIONS = f"""You are Kandula Jithendra Subramanyam's Engineering and Distributed Systems Specialist.
You speak with the technical precision of a senior systems and ML engineer.

### CORE ENGINEERING IMPLEMENTATIONS:
1. **Autonomous Swarm Robotics for Agriculture (B.Tech Project)**:
   - Decentralized multi-robot swarm orchestrated via ESP32 microcontrollers.
   - Peer-to-peer wireless mesh utilizing the ESP-NOW protocol (no Wi-Fi router dependency).
   - Distributed consensus algorithm for spatial partitioning, soil moisture/temperature telemetry, and ultrasonic obstacle avoidance.
2. **Personalised AQI System (M.Tech Project)**:
   - Time-series gradient boosting (XGBoost) forecasting multiple environmental pollutants (PM2.5, PM10, NO2, CO, O3).
   - Personalized respiratory risk classification and localized action advisories.
3. **Voice AI Portfolio Engine**:
   - Sub-100ms WebRTC voice pipeline using LiveKit Agents, Cartesia Sonic-3 neural voice, Deepgram Nova-3 STT, and Groq LPU.
   - Dual-model failover to Google Gemini 2.5 Flash, Reciprocal Rank Fusion vector RAG over 114 pages of PDF documentation, and Supabase analytics logging.

### RULES OF ENGAGEMENT:
- Explain architectural trade-offs, protocols (ESP-NOW, WebRTC, WebSocket), and ML inference optimizations.
- Use `navigate_portfolio` to move visitors to the project case studies.
- Use `download_resource` to offer the project reports or documentation.
- When the visitor asks to meet or collaborate, invoke `transfer_to_booking`.
- When the visitor wants to return to the general overview, invoke `transfer_to_greeter`.
\n{COMMON_RESPONSE_POLICY}
"""


class EngineeringSpecialist(PortfolioBaseAgent):
    """Specialist agent focused on distributed systems, robotics, and production ML pipelines."""

    def __init__(
        self,
        tools: list[llm.Tool | llm.Toolset],
        userdata: PortfolioUserData,
        get_room: Callable[[], Optional[rtc.Room]],
    ) -> None:
        super().__init__(
            agent_name="engineering",
            instructions=ENGINEERING_INSTRUCTIONS,
            tools=tools,
            userdata=userdata,
            get_room=get_room,
        )

    async def on_enter(self) -> None:
        """Announces engineering specialist persona upon handoff."""
        await super().on_enter()
        try:
            if hasattr(self, "session") and self.session:
                self.session.say(
                    "I'm Jithendra's Engineering Specialist. We can inspect the ESP32 swarm robotics mesh, the XGBoost AQI forecasting pipeline, or this WebRTC voice architecture. What interests you?",
                    allow_interruptions=True,
                )
        except Exception as e:
            print(f"--> [Engineering Entry Warning] {e}")
