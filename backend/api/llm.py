import ssl
from typing import Any, Optional
import certifi
import httpx
import openai
from livekit.agents import llm
from livekit.plugins import google, openai as lk_openai

from config import settings

_CACHED_SSL_CONTEXT: Optional[ssl.SSLContext] = None
_CACHED_HTTPX_CLIENT: Optional[httpx.AsyncClient] = None
_CACHED_GROQ_CLIENT: Optional[openai.AsyncClient] = None
_CACHED_GROQ_LLM: Optional[lk_openai.LLM] = None
_CACHED_GEMINI_FALLBACK: Optional[google.LLM] = None


def get_ssl_context() -> ssl.SSLContext:
    """Returns a singleton SSL context loaded with Mozilla Root CA certs in memory."""
    global _CACHED_SSL_CONTEXT
    if _CACHED_SSL_CONTEXT is None:
        _CACHED_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
    return _CACHED_SSL_CONTEXT


def get_httpx_client() -> httpx.AsyncClient:
    """
    Returns a persistent singleton AsyncClient reusing the preloaded SSL context
    and connection pool, eliminating disk I/O and TLS renegotiation delays.
    """
    global _CACHED_HTTPX_CLIENT
    if _CACHED_HTTPX_CLIENT is None or _CACHED_HTTPX_CLIENT.is_closed:
        _CACHED_HTTPX_CLIENT = httpx.AsyncClient(
            verify=get_ssl_context(),
            timeout=httpx.Timeout(connect=10.0, read=15.0, write=5.0, pool=10.0),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=120,
            ),
        )
    return _CACHED_HTTPX_CLIENT


def _sanitize_groq_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """
    Enforces strict OpenAI specification whitelist on all messages.
    Completely eliminates 'property extra_content is unsupported' from Groq
    by dropping any unapproved keys attached by LiveKit or fallback providers.
    """
    clean_messages = []
    allowed_msg_keys = {"role", "content", "name", "tool_calls", "tool_call_id"}
    for msg in messages:
        if isinstance(msg, dict):
            clean = {k: v for k, v in msg.items() if k in allowed_msg_keys}
            if "tool_calls" in clean and isinstance(clean["tool_calls"], list):
                clean_tool_calls = []
                for tc in clean["tool_calls"]:
                    if isinstance(tc, dict):
                        tc_clean = {k: v for k, v in tc.items() if k in {"id", "type", "function"}}
                        clean_tool_calls.append(tc_clean)
                    else:
                        clean_tool_calls.append(tc)
                clean["tool_calls"] = clean_tool_calls
            clean_messages.append(clean)
        else:
            clean_messages.append(msg)
    return clean_messages


def get_groq_client() -> openai.AsyncClient:
    """Returns a singleton OpenAI client pre-configured with Groq and request sanitization."""
    global _CACHED_GROQ_CLIENT
    if _CACHED_GROQ_CLIENT is None:
        client = openai.AsyncClient(
            api_key=settings.GROQ_API_KEY,
            base_url=settings.GROQ_BASE_URL,
            http_client=get_httpx_client(),
            max_retries=0,
        )
        orig_create = client.chat.completions.create

        async def sanitized_create(*args, **kwargs):
            if "messages" in kwargs and isinstance(kwargs["messages"], list):
                kwargs["messages"] = _sanitize_groq_messages(kwargs["messages"])
            return await orig_create(*args, **kwargs)

        client.chat.completions.create = sanitized_create
        _CACHED_GROQ_CLIENT = client
    return _CACHED_GROQ_CLIENT


def build_llm_pipeline() -> llm.LLM:
    """
    Constructs a fault-tolerant, high-performance LLM pipeline:
    1. Primary: Groq LPU with automatic message sanitization (sub-100ms TTFT).
    2. Fallback: Google Gemini 2.5 Flash via the direct Google API.
    Reuses pre-warmed singleton clients to ensure 0ms event loop stall on session init.
    """
    global _CACHED_GROQ_LLM, _CACHED_GEMINI_FALLBACK

    if _CACHED_GEMINI_FALLBACK is None and settings.GOOGLE_API_KEY:
        _CACHED_GEMINI_FALLBACK = google.LLM(
            model=settings.FALLBACK_MODEL,
            api_key=settings.GOOGLE_API_KEY,
            temperature=settings.GROQ_TEMPERATURE,
            max_output_tokens=settings.GROQ_MAX_TOKENS,
        )

    if settings.GROQ_API_KEY:
        if _CACHED_GROQ_LLM is None:
            _CACHED_GROQ_LLM = lk_openai.LLM(
                model=settings.GROQ_MODEL,
                client=get_groq_client(),
                max_completion_tokens=settings.GROQ_MAX_TOKENS,
                temperature=settings.GROQ_TEMPERATURE,
            )
        providers: list[llm.LLM] = [_CACHED_GROQ_LLM]
        if _CACHED_GEMINI_FALLBACK is not None:
            providers.append(_CACHED_GEMINI_FALLBACK)
        return llm.FallbackAdapter(
            providers,
            attempt_timeout=settings.LLM_ATTEMPT_TIMEOUT,
            max_retry_per_llm=settings.LLM_MAX_RETRY,
        )

    if _CACHED_GEMINI_FALLBACK is not None:
        return _CACHED_GEMINI_FALLBACK

    raise RuntimeError("Set GROQ_API_KEY or GOOGLE_API_KEY to start the voice agent.")
