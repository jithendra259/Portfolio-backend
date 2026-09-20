"""
Groq Orpheus TTS wrapper for LiveKit Agents (livekit-agents >= 1.8).

Groq's Orpheus model is limited to 200 characters per request.
This wrapper splits long text into sentence-boundary chunks, synthesises each
chunk via the OpenAI-compatible /audio/speech endpoint, and pushes the raw
WAV bytes into the AudioEmitter — making it transparent to the rest of the
voice pipeline (including tts.FallbackAdapter).
"""

from __future__ import annotations

import re
import ssl
import uuid

import certifi
import httpx
from livekit.agents.tts import (
    TTS,
    AudioEmitter,
    ChunkedStream,
    TTSCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from config import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GROQ_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"
_ORPHEUS_MODEL = "canopylabs/orpheus-v1-english"
_ORPHEUS_VOICE = "daniel"      # calm, professional male voice
_MAX_CHARS = 190               # Stay safely below Groq's 200-char limit
_SAMPLE_RATE = 24_000          # Orpheus outputs 24 kHz wav
_NUM_CHANNELS = 1
_MIME_TYPE = "audio/wav"
_WAV_HEADER_BYTES = 44         # Standard PCM WAV header size


def _split_into_chunks(text: str, max_chars: int = _MAX_CHARS) -> list[str]:
    """
    Split text on sentence boundaries (. ! ?) so each chunk fits within
    max_chars. Falls back to comma-splits, then word-splits.
    """
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            parts = re.split(r"(?<=,)\s+", sentence)
            for part in parts:
                if len(part) > max_chars:
                    for word in part.split():
                        candidate = (current + " " + word).strip()
                        if len(candidate) > max_chars:
                            if current:
                                chunks.append(current)
                            current = word
                        else:
                            current = candidate
                else:
                    candidate = (current + " " + part).strip()
                    if len(candidate) > max_chars:
                        if current:
                            chunks.append(current)
                        current = part
                    else:
                        current = candidate
        else:
            candidate = (current + " " + sentence).strip()
            if len(candidate) > max_chars:
                if current:
                    chunks.append(current)
                current = sentence
            else:
                current = candidate

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


class _OrpheusChunkedStream(ChunkedStream):
    """ChunkedStream implementation for Groq Orpheus."""

    def __init__(
        self,
        tts_instance: "GroqOrpheusTTS",
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts_instance, input_text=input_text, conn_options=conn_options)

    async def _run(self, output_emitter: AudioEmitter) -> None:
        chunks = _split_into_chunks(self._input_text)
        request_id = str(uuid.uuid4())

        # Initialise the emitter once for the whole utterance
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
            mime_type=_MIME_TYPE,
        )

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        async with httpx.AsyncClient(
            verify=ssl_ctx,
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=5.0, pool=10.0),
        ) as client:
            for chunk_text in chunks:
                if not chunk_text:
                    continue

                resp = await client.post(
                    _GROQ_TTS_URL,
                    headers={
                        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": _ORPHEUS_MODEL,
                        "voice": _ORPHEUS_VOICE,
                        "input": chunk_text,
                        "response_format": "wav",
                    },
                )
                resp.raise_for_status()

                # Strip the 44-byte WAV header and push raw PCM bytes
                raw_pcm = resp.content[_WAV_HEADER_BYTES:]
                output_emitter.push(raw_pcm)

        output_emitter.flush()


class GroqOrpheusTTS(TTS):
    """
    LiveKit TTS plugin for Groq Orpheus.

    Drop-in replacement usable inside AgentSession(tts=...) or
    tts.FallbackAdapter([GroqOrpheusTTS(), elevenlabs.TTS(...)]).
    """

    def __init__(self) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return _OrpheusChunkedStream(self, text, conn_options)
