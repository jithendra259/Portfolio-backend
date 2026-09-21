"""
Voice Session Factory for LiveKit Voice Agent.
Uses Groq Whisper for STT, Groq Orpheus TTS (primary) with ElevenLabs fallback,
and Groq LLM with Google Gemini 2.5 Flash as LLM fallback.
"""

from livekit import agents
from livekit.agents import (
    AgentSession,
    TurnHandlingOptions,
    inference,
    text_transforms,
    tts,
)
from livekit.plugins import openai, elevenlabs

from api import build_llm_pipeline, GroqOrpheusTTS
from config import settings
from prompts import PRONUNCIATION_REPLACEMENTS


def create_voice_session(ctx: agents.JobContext | None = None) -> AgentSession:
    """
    Constructs an ultra-low latency, fault-tolerant voice pipeline:

    STT:  Deepgram Nova-3 (primary)
          → AssemblyAI Universal Streaming (automatic fallback adapter on 429/connection error)
          → Deepgram Nova-2 (secondary fallback)

    TTS:  Groq Orpheus (primary, sub-90ms synthesis) → ElevenLabs Multilingual v2 (automatic fallback adapter)

    LLM:  Groq LPU (primary) → Google Gemini 2.5 Flash (agent-side FallbackAdapter)

    Turn: LiveKit Cloud TurnDetector v1 (0% local CPU on Render)
          Adaptive interruption — filters backchannels ("uh-huh", "ok", "right")
          backchannel_boundary=(1.0, 2.0) — extra 2s end-window for Deepgram transcript latency
          Preemptive LLM generation (no preemptive TTS — saves Render CPU)
    """
    # We use Groq's high-rate-limit Whisper endpoint for Speech-to-Text
    stt_pipeline = openai.STT(
        model=settings.STT_MODEL,
        language=settings.STT_LANGUAGE,
        base_url=settings.GROQ_BASE_URL,
        api_key=settings.GROQ_API_KEY,
    )

    if not settings.ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY must be configured as TTS fallback.")

    # Primary TTS: Groq Orpheus — ultra-fast, realistic voice (auto-chunked to ≤190 chars)
    # Fallback TTS: ElevenLabs — kicks in automatically if Orpheus returns an error
    tts_pipeline = tts.FallbackAdapter(
        [
            GroqOrpheusTTS(),
            elevenlabs.TTS(
                model=settings.TTS_MODEL,
                voice_id=settings.TTS_VOICE_ID,
                api_key=settings.ELEVENLABS_API_KEY,
            ),
        ],
        max_retry_per_tts=1,
    )


    return AgentSession(
        stt=stt_pipeline,
        llm=build_llm_pipeline(),
        tts=tts_pipeline,

        # ── Turn handling: detection + adaptive interruption + preemptive gen ────────
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(version=settings.TURN_DETECTOR_VERSION),
            endpointing={
                # min_delay: wait at least 0.5s of silence before confirming end-of-turn
                # max_delay: force close after 3.0s max to avoid indefinite wait
                "min_delay": settings.MIN_ENDPOINTING_DELAY,
                "max_delay": settings.MAX_ENDPOINTING_DELAY,
            },
            interruption={
                # Adaptive mode: LiveKit Cloud inference model distinguishes real barge-ins
                # from conversational backchannels ("uh-huh", "ok", "right", "sure")
                "mode": "adaptive",
                # Min 0.5s of speech required to count as an interruption
                "min_duration": 0.5,
                # No minimum word count — acoustic model alone decides
                "min_words": 0,
                # 2.0s silence after a detected barge-in before classifying as false-positive
                "false_interruption_timeout": 2.0,
                # Resume speaking if interruption was a false-positive (e.g. background noise)
                "resume_false_interruption": True,
                # Cooldown around agent turn boundaries:
                # start=1.0s: use VAD during first second of agent speech (catch early barge-ins)
                # end=2.0s:   include late Deepgram transcripts as real turns (Deepgram can lag ~1.5s)
                "backchannel_boundary": (1.0, 2.0),
            },
            preemptive_generation={
                # Begin LLM generation as soon as final STT transcript arrives,
                # before the turn-detection model confirms end-of-turn → cuts perceived latency
                "enabled": True,
                # preemptive_tts=False: prevents wasted Cartesia synthesis on cancelled turns
                # (Render free tier has 0.1 vCPU — every wasted cycle matters)
                "preemptive_tts": False,
                # Skip preemptive generation for long utterances (>10s) — they mutate too often
                "max_speech_duration": 10.0,
                # Retry up to 3 times per turn if the final transcript keeps changing
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
