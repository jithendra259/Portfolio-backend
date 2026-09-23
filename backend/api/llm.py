import asyncio
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
_CACHED_PROVIDERS: Optional[list[llm.LLM]] = None


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
    """Returns a singleton OpenAI client pre-configured with Groq, request sanitization, and TPM recovery."""
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
            try:
                return await orig_create(*args, **kwargs)
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "rate_limit" in err_str.lower() or "too many requests" in err_str.lower():
                    if "tpd" in err_str.lower() or "tokens per day" in err_str.lower():
                        raise
                    delay = 2.0
                    try:
                        if "try again in " in err_str:
                            part = err_str.split("try again in ")[1].split("s")[0]
                            delay = min(max(float(part) + 0.3, 1.0), 5.0)
                    except Exception:
                        pass
                    await asyncio.sleep(delay)
                    return await orig_create(*args, **kwargs)
                raise

        client.chat.completions.create = sanitized_create
        _CACHED_GROQ_CLIENT = client
    return _CACHED_GROQ_CLIENT


def get_fallback_models() -> list[llm.LLM]:
    """
    Constructs an ultra-resilient multi-tier fallback LLM cascade across all available models:
    1. Primary: Groq Qwen 3.8 27B (sub-100ms ultra-low latency, fresh quota).
    2. Fallback 1: Groq GPT-OSS 120B (high reasoning capacity on Groq LPU).
    3. Fallback 2: Groq GPT-OSS 20B (fast lightweight model on Groq LPU).
    4. Fallback 3: Google Gemini 2.5 Flash (direct Google API).
    5. Fallback 4: Google Gemini 2.5 Pro (complex reasoning fallback).
    """
    global _CACHED_PROVIDERS
    if _CACHED_PROVIDERS is not None:
        return _CACHED_PROVIDERS

    providers: list[llm.LLM] = []

    # 1. Groq LPU Models (Sub-100ms primary & secondary tiers)
    if settings.GROQ_API_KEY:
        groq_client = get_groq_client()
        groq_models = [
            settings.GROQ_MODEL,        # Default: qwen/qwen3.8-27b
            "openai/gpt-oss-120b",      # Fallback: 120B parameter model
            "openai/gpt-oss-20b",       # Fallback: 20B parameter model
        ]
        seen_groq = set()
        for m in groq_models:
            if m and m not in seen_groq:
                seen_groq.add(m)
                providers.append(
                    lk_openai.LLM(
                        model=m,
                        client=groq_client,
                        max_completion_tokens=settings.GROQ_MAX_TOKENS,
                        temperature=settings.GROQ_TEMPERATURE,
                    )
                )

    # 2. Google Gemini Models (Direct Google API fallback tiers)
    if settings.GOOGLE_API_KEY:
        gemini_models = [
            settings.FALLBACK_MODEL,    # Default: gemini-2.5-flash
            "gemini-2.5-pro",           # Fallback: deep analytical model
        ]
        seen_gemini = set()
        for m in gemini_models:
            if m and m not in seen_gemini:
                seen_gemini.add(m)
                providers.append(
                    google.LLM(
                        model=m,
                        api_key=settings.GOOGLE_API_KEY,
                        temperature=settings.GROQ_TEMPERATURE,
                        max_output_tokens=settings.GROQ_MAX_TOKENS,
                    )
                )

    _CACHED_PROVIDERS = providers
    return _CACHED_PROVIDERS


def build_llm_pipeline() -> llm.LLM:
    """
    Constructs a fault-tolerant LLM pipeline with all available models as fallbacks.
    Cascades seamlessly from Groq LPU models to Gemini direct API models.
    """
    providers = get_fallback_models()
    if not providers:
        raise RuntimeError("Set GROQ_API_KEY or GOOGLE_API_KEY to start the voice agent.")

    if len(providers) == 1:
        return providers[0]

    return llm.FallbackAdapter(
        providers,
        attempt_timeout=settings.LLM_ATTEMPT_TIMEOUT,
        max_retry_per_llm=settings.LLM_MAX_RETRY,
    )
