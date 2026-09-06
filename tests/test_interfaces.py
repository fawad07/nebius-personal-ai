"""Structural conformance: each concrete pipeline class satisfies its Protocol."""

import pytest

from src import interfaces
from src.conversation.conversation_manager import ConversationManager  # noqa: F401
from src.llm.llm_client import LLMClient
from src.speaker.speaker_identifier import SpeakerIdentifier
from src.stt.stt_service import STTService
from src.tts.system_tts import SystemTTS


def test_llm_client_is_llm_provider():
    assert issubclass(LLMClient, interfaces.LLMProvider)


def test_stt_service_is_stt_provider():
    assert issubclass(STTService, interfaces.STTProvider)


def test_system_tts_is_tts_provider():
    # SystemTTS is the default engine and torch-free, so this always runs.
    assert issubclass(SystemTTS, interfaces.TTSProvider)


def test_xtts_is_tts_provider():
    # XTTS voice cloning is an optional extra (Coqui TTS + librosa); skip when
    # those aren't installed rather than failing collection.
    pytest.importorskip("TTS")
    pytest.importorskip("librosa")
    from src.tts.voice_cloning_tts import VoiceCloningTTS
    assert issubclass(VoiceCloningTTS, interfaces.TTSProvider)


def test_speaker_identifier_is_speaker_provider():
    assert issubclass(SpeakerIdentifier, interfaces.SpeakerProvider)
