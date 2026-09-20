"""
Route mappings, keyword rules, and navigation targets for LangGraph intent routing.
"""

from prompts.knowledge import NAVIGATION_TARGETS

# Tuple of (route_name, target_screen, keywords)
ROUTE_RULES = (
    (
        "research_eaai",
        "case_study_adaptive_governance",
        ("eaai", "g-cvar", "contagion", "fire sale", "bipartite", "sec 13-f", "eigenvector centrality", "paper 1"),
    ),
    (
        "research_lncs",
        "case_study_regime_supervisory",
        ("lncs", "ijcaci", "instability", "regime", "drift", "ledoit", "shrinkage", "paper 2"),
    ),
    (
        "research_cor",
        "case_study_supervisory_xai",
        (
            "cor",
            "clarabel",
            "convex",
            "xai",
            "dag",
            "grounding",
            "supervisory portfolio",
            "socp",
            "paper 3",
        ),
    ),
    (
        "project_voice_architecture",
        "case_study_voice_architecture",
        ("voice", "webrtc", "livekit", "groq", "cartesia", "deepgram", "latency", "voice agent", "speech"),
    ),
    (
        "project_aqi",
        "case_study_aqi",
        ("aqi", "air quality", "delhi", "pm2.5", "xgboost", "cpcb"),
    ),
    (
        "project_swarm_robotics",
        "case_study_swarm_robotics",
        ("swarm", "robot", "agriculture", "esp32", "kscst", "densenet", "drone"),
    ),
)
