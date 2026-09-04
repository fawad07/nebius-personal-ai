from unittest.mock import patch

from src.stt.stt_service import STTService


@patch("src.stt.stt_service.WhisperModel")
def test_stt_initialization(mock_whisper_model):
    stt = STTService(model_path="models/stt/whisper-medium", use_gpu=False)
    assert stt.model_path == "models/stt/whisper-medium"
    mock_whisper_model.assert_called_once_with("models/stt/whisper-medium", device="cpu")


@patch("src.stt.stt_service.WhisperModel")
def test_stt_transcription_empty(mock_whisper_model):
    mock_whisper_model.return_value.transcribe.return_value = ([], None)

    stt = STTService(model_path="models/stt/whisper-medium", use_gpu=False)
    result = stt.transcribe(b"")
    assert result == ""
