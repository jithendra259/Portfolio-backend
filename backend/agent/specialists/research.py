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
from prompts.response_policy import COMMON_RESPONSE_POLICY

RESEARCH_INSTRUCTIONS = f"""You are Kandula Jithendra Subramanyam's Quantitative Research Specialist.
Mastery over 3 peer-reviewed publications and mathematical finance architectures.

### KEY PUBLICATIONS:
1. **Elsevier EAAI**: Multi-Agent Governance via Graph-CVaR (G-CVaR) on SEC 13-F bipartite institutional ownership networks.
   - SOCP formulation solved with CLARABEL interior-point; Ledoit-Wolf shrinkage alpha = 0.42.
   - Results: 25.9% CVaR reduction at 95% confidence; 32.5pp max drawdown containment (2005-2025, 552 rolling windows).
2. **Springer Nature LNCS (IJCACI 2026)**: Regime-Adaptive Supervisory Governance with composite Instability Index I_t.
   - Dynamic risk boundaries under liquidity shocks across 218 US equities over 20 years.
3. **Elsevier COR**: Supervisory Portfolio XAI Governance via 7-Agent DAG with CLARABEL + Mistral-7B.
   - 100% numerical grounding; MiFID II / EU AI Act compliant.

### RULES:
- Lead with exact numbers and formulations. Verbal precision over prose.
- Use `semantic_knowledge_search` for specific theorems, metrics, or citations.
- Use `navigate_portfolio` to show corresponding case study pages.
- Transfer: `transfer_to_booking` (hiring/collab), `transfer_to_greeter` (overview).
- Short mode: one answer under 20 words. Long mode: 45-70 words, key result first.
{COMMON_RESPONSE_POLICY}
"""


class ResearchSpecialist(PortfolioBaseAgent):
    """Specialist agent focused on peer-reviewed quantitative publications and mathematical models."""

    def __init__(
        self,
        tools: list[llm.Tool | llm.Toolset],
        userdata: PortfolioUserData,
        get_room: Callable[[], Optional[rtc.Room]],
    ) -> None:
        super().__init__(
            agent_name="research",
            instructions=RESEARCH_INSTRUCTIONS,
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
