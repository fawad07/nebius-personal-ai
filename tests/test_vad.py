from unittest.mock import MagicMock, patch

import numpy as np

from src.audio.vad import VoiceActivityDetector


@patch("src.audio.vad.torch.hub.load")
def test_vad_detects_silence(mock_hub_load):
    mock_model = MagicMock()
    mock_model.return_value = MagicMock(item=lambda: 0.1)
    mock_hub_load.return_value = (mock_model, None)

    vad = VoiceActivityDetector()
    silence = np.zeros(512, dtype=np.float32).tobytes()

    assert vad.is_speech(silence) is False


@patch("src.audio.vad.torch.hub.load")
def test_vad_detects_speech(mock_hub_load):
    mock_model = MagicMock()
    mock_model.return_value = MagicMock(item=lambda: 0.9)
    mock_hub_load.return_value = (mock_model, None)

    vad = VoiceActivityDetector()
    chunk = np.ones(512, dtype=np.float32).tobytes()

    assert vad.is_speech(chunk) is True


@patch("src.audio.vad.torch.hub.load")
def test_vad_empty_chunk_is_not_speech(mock_hub_load):
    mock_hub_load.return_value = (MagicMock(), None)

    vad = VoiceActivityDetector()
    assert vad.is_speech(b"") is False
