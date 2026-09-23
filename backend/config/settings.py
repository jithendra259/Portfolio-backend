import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment configurations
BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env.local")
load_dotenv(BACKEND_DIR / ".env")
load_dotenv()


class Settings:
    # Server Networking
    PORT: int = int(os.getenv("PORT", "10000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    # Groq LPU Configuration (Sub-100ms primary LLM, 30,000 ITPM quota)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip("\"' \t\r\n")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    GROQ_MAX_TOKENS: int = 800
    GROQ_TEMPERATURE: float = 0.2

    # Groq TTS (Orpheus) Configuration
    GROQ_TTS_MODEL: str = os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
    GROQ_TTS_VOICE: str = os.getenv("GROQ_TTS_VOICE", "autumn")

    # Deepgram TTS Configuration (Sub-200ms, high reliability, zero 429 rate-limiting)
    DEEPGRAM_TTS_MODEL: str = os.getenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en")

    # Google Gemini fallback (direct Google API, not LiveKit Inference)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "").strip("\"' \t\r\n")
    FALLBACK_MODEL: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
    LLM_ATTEMPT_TIMEOUT: float = 18.0  # Allows auto-backoff token bucket replenishment
    LLM_MAX_RETRY: int = 1

    # Direct provider audio credentials. These bypass LiveKit Inference quotas.
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "").strip("\"' \t\r\n")
    CARTESIA_API_KEY: str = os.getenv("CARTESIA_API_KEY", "").strip("\"' \t\r\n")

    # Speech-To-Text (Deepgram Nova-3 streaming)
    STT_MODEL: str = os.getenv("STT_MODEL", "nova-3")
    STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "en")

    # Turn Detection & Latency Tuning
    # 'v1-mini' runs locally on CPU with 0ms network latency; 'v1' is Cloud inference
    TURN_DETECTOR_VERSION: str = os.getenv("TURN_DETECTOR_VERSION", "v1-mini")
    MIN_ENDPOINTING_DELAY: float = 0.85  # Prevents early turn commits on short pauses
    MAX_ENDPOINTING_DELAY: float = 3.5
    USER_TURN_MAX_WORDS: int = 50
    USER_TURN_MAX_DURATION: float = 25.0
    USER_AWAY_TIMEOUT: float = 25.0
    IDLE_DISCONNECT_TIMEOUT: float = 35.0

    # Dense semantic retrieval loads PyTorch + SentenceTransformers, which is too
    # large for Render Free's 512 MB process limit. The sparse TF-IDF retriever is
    # sufficient for this small, portfolio-specific knowledge base.
    ENABLE_DENSE_RAG: bool = os.getenv("ENABLE_DENSE_RAG", "false").lower() == "true"

    # LiveKit Cloud Credentials
    LIVEKIT_URL: str = os.getenv("LIVEKIT_URL", "")
    LIVEKIT_API_KEY: str = os.getenv("LIVEKIT_API_KEY", "")
    LIVEKIT_API_SECRET: str = os.getenv("LIVEKIT_API_SECRET", "")

    # Supabase Credentials (Analytics, Conversation Turns, Booking Leads)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", os.getenv("NEXT_PUBLIC_SUPABASE_URL", ""))
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", ""))

settings = Settings()
