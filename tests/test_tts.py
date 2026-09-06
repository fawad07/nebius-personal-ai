from unittest.mock import patch

import numpy as np
import pytest

# XTTS voice cloning is an optional extra (Coqui TTS + librosa). Skip this whole
# module when those aren't installed instead of erroring at collection.
pytest.importorskip("TTS")
pytest.importorskip("librosa")

from src.tts.voice_cloning_tts import VoiceCloningTTS


@patch("src.tts.voice_cloning_tts.TTS")
def test_tts_initialization(mock_tts_cls):
    tts = VoiceCloningTTS(
        model_path="models/tts/xtts_model.pth",
        voice_profile_path="data/samples/agent_voice.wav",
    )
    assert tts.model_path == "models/tts/xtts_model.pth"
    assert tts.voice_profile == "data/samples/agent_voice.wav"


@patch("src.tts.voice_cloning_tts.TTS")
def test_tts_synthesize_empty(mock_tts_cls):
    mock_tts_cls.return_value.tts.return_value = np.zeros(0, dtype=np.float32)

    tts = VoiceCloningTTS(
        model_path="models/tts/xtts_model.pth",
        voice_profile_path="data/samples/agent_voice.wav",
    )
    audio = tts.synthesize("", {"mood": "neutral"})
    assert isinstance(audio, bytes)
