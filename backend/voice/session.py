"""
Voice Session Factory for LiveKit Voice Agent.

STT:  Deepgram Flux (primary, built-in turn detection via STT)
      No local VAD — STT handles end-of-turn detection.

TTS:  ElevenLabs Multilingual v2 (primary — reliable, no terms acceptance)

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
    text_transforms,
)
from livekit.plugins import deepgram, elevenlabs

from api import build_llm_pipeline
from config import settings
from prompts import PRONUNCIATION_REPLACEMENTS


def create_voice_session(ctx: agents.JobContext | None = None) -> AgentSession:
    """
    Constructs an ultra-low latency, fault-tolerant voice pipeline
    optimized for Render Free tier (0.1 vCPU / 512 MB):

    STT:  Deepgram Flux (STTv2) with built-in turn detection via STT
          vad=None — no local VAD inference, zero CPU cost for VAD

    TTS:  ElevenLabs Multilingual v2 (primary)

    LLM:  Groq LPU (primary) → Google Gemini 2.5 Flash (fallback)

    Turn: STT-based turn detection via Deepgram Flux
          Adaptive interruption via LiveKit Cloud inference
          Preemptive LLM generation (no preemptive TTS)
    """
    # ── STT: Deepgram Flux with STT-based turn detection ───────────────────────
    # Flux has built-in end-of-turn detection; no local VAD needed.
    # This eliminates the CPU starvation that Silero VAD caused on Render 0.1 vCPU.
    # Note: STTv2 doesn't accept 'language' param; model="flux-general-en" selects English.
    stt_pipeline = deepgram.STTv2(
        model="flux-general-en",
        api_key=settings.DEEPGRAM_API_KEY,
    )

    # ── TTS: ElevenLabs (primary only — Groq Orpheus fallback removed) ─────────
    # LiveKit's ElevenLabs plugin expects ELEVEN_API_KEY env var.
    # model="eleven_multilingual_v2" for quality, "eleven_turbo_v2_5" for lower latency.
    tts_pipeline = elevenlabs.TTS(
        model=settings.TTS_MODEL,
        voice_id=settings.TTS_VOICE_ID,
        api_key=settings.ELEVEN_API_KEY,
    )


    return AgentSession(
        stt=stt_pipeline,
        llm=build_llm_pipeline(),
        tts=tts_pipeline,

        # ── Turn handling: STT-based detection + adaptive interruption ──────────
        turn_handling=TurnHandlingOptions(
            # Use Deepgram Flux's built-in STT turn detection instead of local VAD
            turn_detection="stt",
            endpointing={
                "min_delay": settings.MIN_ENDPOINTING_DELAY,
                "max_delay": settings.MAX_ENDPOINTING_DELAY,
            },
            interruption={
                # Adaptive mode: LiveKit Cloud inference model distinguishes real barge-ins
                # from conversational backchannels ("uh-huh", "ok", "right", "sure")
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
        # Disable TTS-aligned transcript: avoids expensive FFT processing on Render 0.1 vCPU
        use_tts_aligned_transcript=False,
        tts_text_transforms=[
            "filter_emoji",
            "filter_markdown",
            text_transforms.replace(PRONUNCIATION_REPLACEMENTS),
        ],
    )