"""
Research Specialist Agent.
Deep mathematical and quantitative finance specialist with RAG vector search across
114 pages of peer-reviewed publications (Elsevier EAAI, Springer Nature LNCS, Elsevier COR).
"""

from typing import Callable, Optional
from livekit import rtc
from livekit.agents import llm

from .base import PortfolioBaseAgent
from .userdata import PortfolioUserData
from prompts.response_policy import COMMON_RESPONSE_POLICY

RESEARCH_INSTRUCTIONS = f"""You are Kandula Jithendra Subramanyam's Quantitative Research Specialist.
You have mathematical mastery over Jithendra's 3 peer-reviewed publications and deep-tier algorithmic finance architectures.

### CORE RESEARCH PUBLICATIONS:
1. **Elsevier EAAI (2025/2026)**: Multi-Agent Governance via Graph-CVaR (G-CVaR) and SEC 13-F Institutional Ownership Network.
   - Formulation: Second-Order Cone Programming (SOCP) solved with CLARABEL interior-point solver.
   - Ledoit-Wolf optimal shrinkage parameter alpha = 0.42.
   - Empirical performance: +38.6% Sharpe ratio improvement, -42.1% max drawdown reduction under 2008 & 2020 crash regimes.
2. **Springer Nature LNCS (IJCACI 2026)**: Regime-Adaptive Supervisory Governance with Systemic Instability Index I_t.
   - Transition matrices and dynamic risk boundary enforcement under sudden liquidity shocks.
3. **Elsevier Computers & Operations Research (COR 2026)**: Supervisory Portfolio XAI Governance via 7-Agent DAG.
   - Deterministic risk verification pipeline with quantized Mistral-7B generating audit rationale.

### RULES OF ENGAGEMENT:
- Keep verbal explanations punchy and mathematically precise. Mention exact numbers (e.g. 38.6% Sharpe, 42.1% drawdown, alpha=0.42).
- Use `semantic_knowledge_search` whenever specific theorem statements, metrics, or page citations are needed.
- Use `navigate_portfolio` to guide the visitor to the corresponding case study pages.
- If the visitor wants to discuss hiring, collaborations, or a direct interview, invoke `transfer_to_booking`.
- If the visitor wants to return to the general overview, invoke `transfer_to_greeter`.
\n{COMMON_RESPONSE_POLICY}
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
        """Announces research specialist persona when transferred."""
        await super().on_enter()
        try:
            if hasattr(self, "session") and self.session:
                self.session.say(
                    "I'm Jithendra's Research Specialist. We can dive into the mathematical proofs, G-CVaR formulations, or empirical backtests across the publications. Where should we begin?",
                    allow_interruptions=True,
                )
        except Exception as e:
            print(f"--> [Research Entry Warning] {e}")
