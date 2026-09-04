"""
Provider interfaces for the voice-agent pipeline.

These `Protocol`s define the structural contract each pipeline stage must
satisfy, so an implementation can be swapped (a local LLM for OpenAI, a
different STT/TTS backend, a learned SER model for the acoustic heuristic)
without touching the orchestrator. Protocols are duck-typed: the existing
concrete classes already conform, so no inheritance is required — they exist
to document the seam and to type-check wiring.

`TTSProvider.synthesize` intentionally returns ``(audio, sample_rate)`` so the
output stage — not each TTS backend — owns resampling to the playback rate.
Today's `VoiceCloningTTS` resamples internally; new backends should prefer
returning their native rate and letting the caller resample.
"""

from __future__ import annotations

from typing import Iterator, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class STTProvider(Protocol):
    def transcribe(self, audio_bytes: bytes, beam_size: int = 5) -> str: ...


@runtime_checkable
class LLMProvider(Protocol):
    def generate_text(self, system_prompt: str, user_prompt: str) -> str: ...
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict: ...
    def stream_text(self, system_prompt: str, user_prompt: str) -> Iterator[str]: ...


@runtime_checkable
class TTSProvider(Protocol):
    # Returns raw float32 audio bytes ready for the output stream. See module
    # docstring on the (audio, sample_rate) direction for future backends.
    def synthesize(self, text: str, mood: dict) -> bytes: ...


@runtime_checkable
class EmotionProvider(Protocol):
    # Returns {"label": str, "confidence": float}.
    def classify(self, audio_bytes: bytes) -> dict: ...


@runtime_checkable
class VADProvider(Protocol):
    def is_speech(self, audio_chunk: bytes) -> bool: ...
    def reset(self) -> None: ...


@runtime_checkable
class SpeakerEmbedder(Protocol):
    def embed(self, audio_bytes: bytes) -> np.ndarray: ...


@runtime_checkable
class SpeakerProvider(Protocol):
    # `identify_detailed` returns a result exposing .group/.name/.score/.embedding
    # (see src/speaker/speaker_identifier.py::SpeakerIdentification).
    def identify(self, audio_bytes: bytes) -> str: ...
    def identify_detailed(self, audio_bytes: bytes): ...
