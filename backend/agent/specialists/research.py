"""
Research Specialist Agent.
Deep mathematical and quantitative finance specialist with RAG vector search across
peer-reviewed publications (Elsevier EAAI, Springer Nature LNCS, Elsevier COR).
"""

from typing import Callable, Optional
from livekit import rtc
from livekit.agents import llm

from .base import PortfolioBaseAgent
from .userdata import PortfolioUserData
from prompts import SYSTEM_INSTRUCTIONS, get_specialist_instructions

RESEARCH_ROLE_INSTRUCTIONS = """
### RESEARCH SPECIALIST ROLE:
1. Mastery over 3 peer-reviewed publications (EAAI G-CVaR, LNCS Regime-Adaptive, COR 7-Agent XAI).
2. Lead with exact numbers and formulations. Verbal precision over prose.
3. Use `semantic_knowledge_search` for specific theorems, metrics, or citations.
4. Use `navigate_portfolio` to show corresponding case study pages.
5. Transfer: `transfer_to_booking` (hiring/collab), `transfer_to_greeter` (overview).
6. Short mode: <20 words. Long mode: 45-70 words, key result first.
"""


class ResearchSpecialist(PortfolioBaseAgent):
    """Specialist agent focused on peer-reviewed quantitative publications and mathematical models."""

    def __init__(
        self,
        tools: list[llm.Tool | llm.Toolset],
        userdata: PortfolioUserData,
        get_room: Callable[[], Optional[rtc.Room]],
    ) -> None:
        instructions = SYSTEM_INSTRUCTIONS + RESEARCH_ROLE_INSTRUCTIONS
        super().__init__(
            agent_name="research",
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
                    "Research Specialist ready. We can dive into G-CVaR formulations, regime-adaptive governance, or the 100% grounded XAI framework. Where shall we start?",
                    allow_interruptions=True,
                )
        except Exception as e:
            print(f"--> [Research Entry Warning] {e}")