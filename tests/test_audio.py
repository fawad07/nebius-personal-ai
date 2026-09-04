import numpy as np
import pytest

from exceptions.audio_err import AudioStreamNotStarted
from src.audio.input import AudioInput
from src.audio.output import AudioOutput
from src.audio.stream_config import AudioStreamConfig


def test_stream_config_defaults():
    cfg = AudioStreamConfig()
    assert cfg.rate == 16000
    assert cfg.channels == 1
    assert cfg.chunk_size == 4096
    assert cfg.device_index is None


def test_capture_before_start_raises():
    audio_in = AudioInput(AudioStreamConfig())
    with pytest.raises(AudioStreamNotStarted):
        audio_in.capture_chunk()


def test_play_before_start_raises():
    audio_out = AudioOutput(AudioStreamConfig())
    with pytest.raises(AudioStreamNotStarted):
        audio_out.play(b"")


def test_capture_chunk_returns_float32_bytes(monkeypatch):
    """capture_chunk should serialize the stream's float32 frame to bytes."""
    cfg = AudioStreamConfig(chunk_size=8)
    audio_in = AudioInput(cfg)

    frame = np.ones(8, dtype=np.float32)

    class FakeStream:
        def read(self, n):
            return frame, False

    audio_in.stream = FakeStream()
    out = audio_in.capture_chunk()

    assert isinstance(out, bytes)
    assert np.array_equal(np.frombuffer(out, dtype=np.float32), frame)
