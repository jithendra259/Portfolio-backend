"""
High-Performance In-Memory Knowledge and Response Cache.
Provides dual-mode (Short Description vs Long Description) grounding,
sub-millisecond LRU query caching, and minimal prompt context sizing.
"""

from collections import OrderedDict
import re
import threading
from typing import Any, Optional


# ── Detailed Pre-compiled Knowledge Store (Zero-Latency Retrieval) ───────────

KNOWLEDGE_STORE: dict[str, dict[str, str]] = {
    "profile": {
        "short": (
            "Jithendra is an AI/ML Researcher and Systems Engineer with 3 peer-reviewed papers "
            "in Elsevier and Springer, specializing in convex optimization and voice AI."
        ),
        "long": (
            "Jithendra is an AI/ML Researcher and Systems Engineer with an M.Tech in AI from Somaiya Vidyavihar "
            "(CGPA 8.06) and B.Tech in ECE from Presidency University (CGPA 7.77). He has authored 3 peer-reviewed "
            "papers across Elsevier EAAI, Springer Nature LNCS, and Elsevier COR, focusing on convex conic optimization, "
            "graph regularized contagion governance, and low-latency voice AI systems."
        ),
    },
    "education": {
        "short": (
            "M.Tech in AI from Somaiya Vidyavihar (CGPA 8.06) and B.Tech in ECE from Presidency University (CGPA 7.77, GATE 2024 DA)."
        ),
        "long": (
            "Jithendra holds an M.Tech in Artificial Intelligence & Data Science from Somaiya Vidyavihar University "
            "with an 8.06 CGPA, and a B.Tech in Electronics & Communication from Presidency University with a 7.77 CGPA, "
            "where he earned a Karnataka State Government research grant. He also qualified the GATE 2024 exam in Data Science & AI."
        ),
    },
    "skills": {
        "short": (
            "Core stack includes Python, PyTorch, LangGraph, Conic Optimization (CVXPY/CLARABEL), WebRTC, and FastAPI."
        ),
        "long": (
            "Proficient in Python, PyTorch, and LangGraph for multi-agent workflows; second-order cone programming via "
            "CVXPY and the CLARABEL interior-point solver; real-time streaming with LiveKit WebRTC and Deepgram; plus "
            "TypeScript, Next.js 15, Docker, and PostgreSQL for robust production deployment."
        ),
    },
    "experience": {
        "short": (
            "Applied AI Engineer at Appfabs, developing agentic LLM pipelines and low-latency computer vision systems."
        ),
        "long": (
            "Currently an Applied AI Engineer at Appfabs, architecting agentic multi-agent systems, real-time voice interfaces, "
            "and low-latency vision workflows. Previously spearheaded collegiate robotics research and developed precision agriculture "
            "swarms funded by state government grants."
        ),
    },
    "research_eaai": {
        "short": (
            "Elsevier EAAI paper introducing G-CVaR to penalize SEC 13-F institutional fire-sale contagion via conic SOCP."
        ),
        "long": (
            "Published in Elsevier EAAI, this paper presents Graph-CVaR (G-CVaR) within a 5-agent blackboard system. "
            "It extracts bipartite institutional networks from SEC 13-F filings to penalize fire-sale contagion inside a "
            "CLARABEL Second-Order Cone Programming solver. Across the 2008 and 2020 crash regimes, it yielded a 38.6% Sharpe "
            "improvement and 42.1% max drawdown reduction."
        ),
    },
    "research_lncs": {
        "short": (
            "Springer Nature LNCS paper introducing Composite Instability Index I_t for dynamic Ledoit-Wolf shrinkage (alpha=0.42)."
        ),
        "long": (
            "Published in Springer Nature LNCS, this study develops a composite instability index I_t tracking Frobenius covariance "
            "drift, rolling volatility, and pairwise correlation. It dynamically tunes Ledoit-Wolf shrinkage intensity with "
            "alpha equals 0.42 across 3 market regimes, containing 20-year maximum drawdown to 32.5%."
        ),
    },
    "research_cor": {
        "short": (
            "Elsevier COR paper featuring a 7-agent supervisory DAG ensuring 100% numerically grounded, audit-ready XAI explanations."
        ),
        "long": (
            "Published in Elsevier Computers & Operations Research, this research formulates a 7-agent DAG governance architecture. "
            "By strictly bounding conversational Mistral-7B explanations to CLARABEL interior-point conic solver outputs, "
            "it mathematically guarantees 100% numerical grounding with 0% hallucination, complying with MiFID II and EU AI Act standards."
        ),
    },
    "project_voice_architecture": {
        "short": (
            "Sub-100ms voice AI architecture using LiveKit WebRTC, Groq LPU, Deepgram Nova-3, and Cartesia Sonic-3."
        ),
        "long": (
            "A high-performance voice engineering architecture featuring LiveKit WebRTC, Groq LPU primary inference under 90ms TTFT, "
            "Deepgram Nova-3 speech recognition, and Cartesia Sonic-3 neural voice. It streams bi-directional UI navigation commands "
            "over WebRTC data channels and is optimized for zero loop blocking on Render's 0.1 vCPU."
        ),
    },
    "project_aqi": {
        "short": (
            "Machine learning system forecasting 48-hour Delhi PM2.5 levels using XGBoost with an R-squared of 0.912."
        ),
        "long": (
            "Evaluated XGBoost, Markov chains, and ARIMA on CPCB sensor data across 10 Delhi monitoring stations. XGBoost achieved "
            "top accuracy with an R-squared of 0.912 and an RMSE of 18.4 micrograms per cubic meter for 48-hour ahead PM2.5 forecasting, "
            "serving predictions via Next.js and Flask."
        ),
    },
    "project_swarm_robotics": {
        "short": (
            "Autonomous ESP-NOW agricultural robot swarm with DenseNet121 edge vision, achieving 98.4% field coverage."
        ),
        "long": (
            "Distributed IoT hardware-software swarm robotics platform for precision agriculture powered by ESP32 microcontrollers "
            "over ESP-NOW mesh networking. Edge cameras run DenseNet121 achieving 96.8% crop disease classification accuracy "
            "and 98.4% field coverage. Awarded funding by the Karnataka State Council for Science & Technology."
        ),
    },
    "contact": {
        "short": (
            "Reach Jithendra directly at kandulajithendrasubramanyam@gmail.com or book a virtual meeting on this portfolio."
        ),
        "long": (
            "You can contact Jithendra via email at kandulajithendrasubramanyam@gmail.com, view his code on GitHub, connect on LinkedIn, "
            "or click the booking option to schedule a 30-minute virtual meeting directly on this site."
        ),
    },
    "booking": {
        "short": (
            "Navigating to the interactive appointment booking screen to schedule a 30-minute meeting with Jithendra."
        ),
        "long": (
            "Opening the dedicated interactive calendar page (/book-appointment) where you can select a convenient 30-minute "
            "time slot for an interview, research discussion, or technical collaboration with Jithendra."
        ),
    },
}

# ── Query Mode Detection Patterns ─────────────────────────────────────────────

LONG_DESCRIPTION_PATTERNS = (
    "in detail",
    "in-depth",
    "in depth",
    "detailed",
    "deep dive",
    "deep-dive",
    "tell me more",
    "tell more",
    "elaborate",
    "breakdown",
    "break down",
    "walk me through",
    "how does that work",
    "mathematical formulation",
    "math behind",
    "comprehensive",
    "step by step",
    "thoroughly",
)



def detect_description_mode(query: str) -> str:
    """
    Detects whether the user desires a concise conversational summary ('short')
    or a deep technical breakdown ('long').
    """
    query_lower = query.strip().lower()
    if any(p in query_lower for p in LONG_DESCRIPTION_PATTERNS):
        return "long"
    return "short"


# ── LRU In-Memory Knowledge Cache ─────────────────────────────────────────────

class LRUKnowledgeCache:
    """
    High-performance thread-safe LRU cache for query classifications and syntheses.
    Yields sub-0.05ms hits for recurring or semantically equivalent questions.
    """

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def _make_key(self, query: str, mode: str) -> str:
        clean = re.sub(r"[^\w\s]", "", query.lower()).strip()
        tokens = " ".join(clean.split()[:12])  # normalize to first 12 tokens
        return f"{tokens}:{mode}"

    def get(self, query: str, mode: str) -> Optional[dict[str, Any]]:
        key = self._make_key(query, mode)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, query: str, mode: str, data: dict[str, Any]) -> None:
        key = self._make_key(query, mode)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = data
            if len(self._cache) > self.capacity:
                self._cache.popitem(last=False)

    def get_grounding(self, topic: str, mode: str) -> Optional[str]:
        """Direct lookup from pre-compiled KNOWLEDGE_STORE."""
        entry = KNOWLEDGE_STORE.get(topic)
        if entry:
            return entry.get(mode) or entry.get("short")
        return None


# Global Knowledge Cache Singleton
KNOWLEDGE_CACHE = LRUKnowledgeCache(capacity=512)
