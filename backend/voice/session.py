"""
Voice Session Factory for LiveKit Voice Agent.

STT:  Deepgram Flux (primary, built-in turn detection via STT)
      No local VAD — STT handles end-of-turn detection.

TTS:  Groq Orpheus (primary — official LiveKit plugin)

LLM:  Groq LPU (primary) → Google Gemini 2.5 Flash (agent-side FallbackAdapter)

Turn: Deepgram Flux STT turn detection (0% local CPU on Render)
      No Silero VAD — eliminates CPU starvation on Render Free tier.
      Adaptive interruption via LiveKit Cloud inference.
      backchannel_boundary=(1.0, 2.0) — extra 2s end-window for Deepgram transcript latency.
      Preemptive LLM generation (no preemptive TTS — saves Render CPU).
"""

from livekit import agents
from livekit.agents import (
    AgentSession,
    TurnHandlingOptions,
    inference,
    text_transforms,
    tts,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    AudioConfig,
)
from livekit.plugins import deepgram, groq

from api import build_llm_pipeline
from config import settings
from prompts import PRONUNCIATION_REPLACEMENTS


def create_voice_session(ctx: agents.JobContext | None = None) -> AgentSession:
    """
    Constructs an ultra-low latency, fault-tolerant voice pipeline
    optimized for Render Free tier (0.1 vCPU / 512 MB):

    STT:  Deepgram Flux (STTv2) with built-in turn detection via STT
          vad=None — no local VAD inference, zero CPU cost for VAD

    TTS:  Groq Orpheus (primary — official livekit-plugins-groq)

    LLM:  Groq LPU (primary) → Google Gemini 2.5 Flash (fallback)

    Turn: STT-based turn detection via Deepgram Flux
          Adaptive interruption via LiveKit Cloud inference
          Preemptive LLM generation (no preemptive TTS)
    """
    # ── STT: Deepgram Nova-3 (official LiveKit streaming STT with interim results) ──
    stt_pipeline = deepgram.STT(
        model="nova-3",
        language=settings.STT_LANGUAGE if settings.STT_LANGUAGE != "en" else "en-US",
        api_key=settings.DEEPGRAM_API_KEY,
        smart_format=True,
        punctuate=True,
        interim_results=True,
    )

    # ── TTS: Deepgram Aura (primary, fast & reliable) -> Groq Orpheus (fallback) ───
    tts_candidates = []
    if settings.DEEPGRAM_API_KEY:
        tts_candidates.append(
            deepgram.TTS(
                model=settings.DEEPGRAM_TTS_MODEL,
                api_key=settings.DEEPGRAM_API_KEY,
            )
        )
    if settings.GROQ_API_KEY:
        tts_candidates.append(
            groq.TTS(
                model=settings.GROQ_TTS_MODEL,
                voice=settings.GROQ_TTS_VOICE,
            )
        )

    if len(tts_candidates) > 1:
        tts_pipeline = tts.FallbackAdapter(tts=tts_candidates, max_retry_per_tts=1)
    elif len(tts_candidates) == 1:
        tts_pipeline = tts_candidates[0]
    else:
        tts_pipeline = deepgram.TTS()

    # ── Background Audio: Thinking sounds during tool calls ─────────────────────────
    background_audio = BackgroundAudioPlayer(
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.6, probability=0.5),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.5, probability=0.3),
        ],
    )

    session = AgentSession(
        stt=stt_pipeline,
        llm=build_llm_pipeline(),
        tts=tts_pipeline,

        # ── Turn handling: Official LiveKit Cloud TurnDetector ───────────────────
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(version=settings.TURN_DETECTOR_VERSION),
            endpointing={
                "min_delay": settings.MIN_ENDPOINTING_DELAY,
                "max_delay": settings.MAX_ENDPOINTING_DELAY,
            },
            interruption={
                "mode": "adaptive",
                "min_duration": 0.5,
                "min_words": 0,
                "false_interruption_timeout": 2.0,
                "resume_false_interruption": True,
                "backchannel_boundary": (1.0, 2.0),
            },
            preemptive_generation={
                "enabled": True,
                "preemptive_tts": False,
                "max_speech_duration": 10.0,
                "max_retries": 3,
            },
            user_turn_limit={
                "max_words": settings.USER_TURN_MAX_WORDS,
                "max_duration": settings.USER_TURN_MAX_DURATION,
            },
        ),
        user_away_timeout=settings.USER_AWAY_TIMEOUT,
        use_tts_aligned_transcript=False,
        tts_text_transforms=[
            "filter_emoji",
            "filter_markdown",
            text_transforms.replace(PRONUNCIATION_REPLACEMENTS),
        ],
    )

    # Start background audio player (thinking sounds during tool calls)
    if ctx:
        import asyncio
        asyncio.create_task(background_audio.start(room=ctx.room, agent_session=session))

    return session