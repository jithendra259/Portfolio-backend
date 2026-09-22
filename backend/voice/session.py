"""
Voice Session Factory for LiveKit Voice Agent.

STT:  Groq Whisper Large V3 (primary, OpenAI-compatible API)
      → Deepgram Nova-3 (fallback, requires DEEPGRAM_API_KEY)
      → AssemblyAI (fallback, requires ASSEMBLYAI_API_KEY)

TTS:  ElevenLabs Multilingual v2 (primary — reliable, no terms acceptance)
      → Groq Orpheus (fallback, requires terms acceptance at console.groq.com)

LLM:  Groq LPU (primary) → Google Gemini 2.5 Flash (agent-side FallbackAdapter)

Turn: LiveKit Cloud TurnDetector v1 (0% local CPU on Render)
      Adaptive interruption — filters backchannels ("uh-huh", "ok", "right")
      backchannel_boundary=(1.0, 2.0) — extra 2s end-window for Deepgram transcript latency
      Preemptive LLM generation (no preemptive TTS — saves Render CPU)
"""

from livekit import agents
from livekit.agents import (
    AgentSession,
    TurnHandlingOptions,
    inference,
    stt,
    text_transforms,
    tts,
)
from livekit.plugins import deepgram, elevenlabs, openai

from api import build_llm_pipeline, GroqOrpheusTTS
from config import settings
from prompts import PRONUNCIATION_REPLACEMENTS


def create_voice_session(ctx: agents.JobContext | None = None) -> AgentSession:
    """
    Constructs an ultra-low latency, fault-tolerant voice pipeline:

    STT:  Groq Whisper Large V3 (primary, OpenAI-compatible API)
          → Deepgram Nova-3 (fallback, requires DEEPGRAM_API_KEY)

    TTS:  ElevenLabs Multilingual v2 (primary — reliable, no terms acceptance)
          → Groq Orpheus (fallback, requires terms acceptance at console.groq.com)

    LLM:  Groq LPU (primary) → Google Gemini 2.5 Flash (agent-side FallbackAdapter)

    Turn: LiveKit Cloud TurnDetector v1 (0% local CPU on Render)
          Adaptive interruption — filters backchannels ("uh-huh", "ok", "right")
          backchannel_boundary=(1.0, 2.0) — extra 2s end-window for Deepgram transcript latency
          Preemptive LLM generation (no preemptive TTS — saves Render CPU)
    """
    # ── STT Pipeline with Fallback ─────────────────────────────────────────────
    stt_providers = [
        openai.STT(
            model=settings.STT_MODEL,
            language=settings.STT_LANGUAGE,
            base_url=settings.GROQ_BASE_URL,
            api_key=settings.GROQ_API_KEY,
        )
    ]

    if settings.DEEPGRAM_API_KEY and settings.DEEPGRAM_API_KEY != "your-deepgram-key":
        stt_providers.append(
            deepgram.STT(
                model="nova-3",
                language=settings.STT_LANGUAGE,
                api_key=settings.DEEPGRAM_API_KEY,
            )
        )

    stt_pipeline = stt.FallbackAdapter(stt_providers) if len(stt_providers) > 1 else stt_providers[0]

    # ── TTS Pipeline with Fallback ─────────────────────────────────────────────
    # ElevenLabs first (reliable, no terms acceptance needed)
    tts_providers = [
        elevenlabs.TTS(
            model=settings.TTS_MODEL,
            voice_id=settings.TTS_VOICE_ID,
            api_key=settings.ELEVENLABS_API_KEY,
        )
    ]
    # Groq Orpheus as fallback (requires terms acceptance at console.groq.com)
    tts_providers.append(GroqOrpheusTTS())

    tts_pipeline = tts.FallbackAdapter(tts_providers, max_retry_per_tts=1)


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
