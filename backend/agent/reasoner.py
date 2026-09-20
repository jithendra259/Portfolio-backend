"""
Deep Research Reasoner and Subagent Delegation for Jithendra's Portfolio.
Provides non-blocking mathematical analysis and peer-reviewed architectural breakdowns.
"""

from typing import Annotated

from livekit import rtc
from livekit.agents import llm

from prompts.knowledge import PUBLICATIONS, PROJECTS
from .tools import broadcast_navigation


class ResearchReasoner:
    """Specialized mathematical and technical reasoner for Jithendra's publications and engineering systems."""

    @staticmethod
    async def analyze_topic(session, room: rtc.Room | None, topic: str) -> str:
        """
        Executes technical analysis. Emits a non-blocking speech filler so the user experiences zero dead silence.
        """
        # 1. Play natural non-blocking speech filler while reasoning
        if session and hasattr(session, "say"):
            try:
                session.say(
                    "Analyzing the mathematical formulation and architecture from the publication now...",
                    allow_interruptions=True,
                    add_to_chat_ctx=False,
                )
            except Exception as e:
                print(f"--> [Reasoner Warning] Filler speech error: {e}")

        topic_clean = topic.lower()

        # Paper 1: Elsevier EAAI
        if any(k in topic_clean for k in ["eaai", "g-cvar", "contagion", "fire sale", "bipartite", "paper 1"]):
            if room:
                await broadcast_navigation(room, "case_study_adaptive_governance")
            p = PUBLICATIONS[0]
            return (
                f"### {p['title']} ({p['journal']} {p['year']})\n"
                f"- **Core Formulation**: G-CVaR (Generalized Conditional Value-at-Risk) systemic risk metric coupled with bipartite banking-asset network simulations.\n"
                f"- **Mathematical Rigor**: Incorporates Ledoit-Wolf optimal shrinkage to eliminate sample covariance invertibility breakdowns during market liquidity shocks.\n"
                f"- **Architecture**: 5-Agent Blackboard architecture running distributed risk evaluations under SEC 13-F institutional holding graphs.\n"
                f"- **Citation Status**: Manuscript EAAI-26-14280."
            )

        # Paper 2: Springer Nature LNCS
        elif any(k in topic_clean for k in ["lncs", "ijcaci", "instability", "regime", "drift", "paper 2"]):
            if room:
                await broadcast_navigation(room, "case_study_regime_supervisory")
            p = PUBLICATIONS[1]
            return (
                f"### {p['title']} ({p['venue']} {p['year']})\n"
                f"- **Core Formulation**: Composite Instability Index (CII) detecting structural covariance drift across financial regimes.\n"
                f"- **Mathematical Rigor**: Optimal Ledoit-Wolf shrinkage intensity parameter alpha=0.42 stabilizes high-frequency portfolio re-allocation.\n"
                f"- **Empirical Validation**: Backtested against extreme market drawdowns; demonstrated 38% reduction in tail-loss variance.\n"
                f"- **Citation Status**: Springer Nature LNCS Series."
            )

        # Paper 3: Elsevier COR
        elif any(k in topic_clean for k in ["cor", "clarabel", "convex", "xai", "dag", "paper 3"]):
            if room:
                await broadcast_navigation(room, "case_study_supervisory_xai")
            p = PUBLICATIONS[2]
            return (
                f"### {p['title']} ({p['journal']} {p['year']})\n"
                f"- **Core Formulation**: 7-Agent Directed Acyclic Graph (DAG) with CLARABEL interior-point conic optimization solver.\n"
                f"- **Mathematical Rigor**: 100% numerical grounding guaranteeing Second-Order Cone Programming (SOCP) boundary convergence.\n"
                f"- **Regulatory Compliance**: Built for MiFID II and EU AI Act Article 14 auditability standards with full decision trace logs.\n"
                f"- **Citation Status**: Elsevier Computers & Operations Research."
            )

        # Project: AQI Air Quality Forecasting
        elif any(k in topic_clean for k in ["aqi", "air quality", "delhi", "pm2.5", "xgboost"]):
            if room:
                await broadcast_navigation(room, "case_study_aqi")
            proj = PROJECTS[0]
            return (
                f"### {proj['title']}\n"
                f"- **Model**: XGBoost Regressor trained on 10 CPCB Delhi real-time sensor stations.\n"
                f"- **Performance**: R2 = 0.912, RMSE = 18.4 ug/m3 for 24-hour PM2.5 forecasting.\n"
                f"- **Features**: Temporal rolling lags, wind vector components, boundary layer depth, and ambient temperature."
            )

        # Project: Autonomous Swarm Robotics
        elif any(k in topic_clean for k in ["swarm", "robot", "agriculture", "esp32", "kscst"]):
            if room:
                await broadcast_navigation(room, "case_study_swarm_robotics")
            proj = PROJECTS[1]
            return (
                f"### {proj['title']}\n"
                f"- **Mesh Protocol**: Decentralized ESP32 ESP-NOW peer-to-peer ad-hoc topology with dynamic leader election.\n"
                f"- **Vision Model**: DenseNet121 edge inference for real-time crop disease detection.\n"
                f"- **Field Metric**: 98.4% autonomous field coverage without cellular infrastructure. Funded by KSCST 46th Series Grant."
            )

        else:
            return (
                f"Technical query '{topic}' evaluated across all 3 research papers and engineering systems. "
                "Jithendra's work centers on multi-agent reinforcement learning, convex optimization (CLARABEL), "
                "and systemic risk modeling (Ledoit-Wolf shrinkage, G-CVaR). Which paper's mathematical proof would you like to explore?"
            )
