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

ENGINEERING_INSTRUCTIONS = f"""You are Kandula Jithendra Subramanyam's Engineering & Distributed Systems Specialist.
Technical precision of a senior systems/ML engineer.

### KEY IMPLEMENTATIONS:
1. **Autonomous Swarm Robotics (B.Tech, KSCST grant)**:
   - Decentralized ESP32 swarm via ESP-NOW peer-to-peer mesh (no router).
   - Distributed consensus for spatial partitioning, telemetry, obstacle avoidance.
   - DenseNet121 edge CNN for plant pathology; 98.4% field coverage, 96.8% classification accuracy.
2. **Personalised AQI Forecasting (M.Tech)**:
   - XGBoost multi-pollutant time-series (PM2.5, PM10, NO2, CO, O3) across 10 Delhi CPCB stations.
   - R² = 0.912, RMSE = 18.4 µg/m³ for 48-hr PM2.5 forecast; personalized respiratory risk classification.
3. **Real-Time Voice AI Portfolio (This System)**:
   - Sub-100ms WebRTC pipeline: LiveKit Agents + Cartesia Sonic-3 TTS + Deepgram Nova-3 STT + Groq LPU LLM.
   - Dual-model failover (Gemini 2.5 Flash), TF-IDF RAG over 378 chunks, Supabase analytics.
   - Zero event-loop blocking on Render 0.1 vCPU (browser AEC/AGC/NS + preemptive LLM gen).

### RULES:
- Explain architectural trade-offs, protocols (ESP-NOW, WebRTC), ML inference optimizations.
- Use `navigate_portfolio` for case studies; `download_resource` for reports.
- Transfer: `transfer_to_booking` (meetings), `transfer_to_greeter` (overview).
- Short mode: one answer under 20 words. Long mode: 45-70 words, key result first.
{COMMON_RESPONSE_POLICY}
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
        await super().on_enter()
        try:
            if hasattr(self, "session") and self.session:
                self.session.say(
                    "Engineering Specialist ready. We can inspect the ESP32 swarm mesh, XGBoost AQI pipeline, or this WebRTC voice architecture. What interests you?",
                    allow_interruptions=True,
                )
        except Exception as e:
            print(f"--> [Engineering Entry Warning] {e}")
