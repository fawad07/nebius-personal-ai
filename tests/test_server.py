import queue

import numpy as np
import pytest

from src.audio.stream_config import AudioStreamConfig
from src.server.audio_bridge import WebSocketAudioInput, WebSocketAudioOutput
from src.server.protocol import decode, encode


def test_protocol_roundtrip():
    msg = decode(encode("state", state="thinking"))
    assert msg == {"type": "state", "state": "thinking"}


def test_protocol_rejects_untyped():
    with pytest.raises(ValueError):
        decode('{"no":"type"}')


def test_input_rechunks_to_config_chunk_size():
    cfg = AudioStreamConfig(chunk_size=4)  # 4 samples * 4 bytes = 16 bytes/chunk
    q = queue.Queue()
    ai = WebSocketAudioInput(cfg, q, poll_timeout=0.01)
    # Push 6 samples across two odd-sized frames; expect a clean 4-sample chunk.
    q.put(np.arange(3, dtype=np.float32).tobytes())
    q.put(np.arange(3, 6, dtype=np.float32).tobytes())
    chunk = ai.capture_chunk()
    assert len(chunk) == 16
    assert np.array_equal(np.frombuffer(chunk, np.float32), np.arange(4, dtype=np.float32))
    # The remaining 2 samples stay buffered, padded to a full chunk next time.
    nxt = ai.capture_chunk()
    assert len(nxt) == 16
    assert np.array_equal(np.frombuffer(nxt, np.float32)[:2], np.array([4, 5], np.float32))


def test_input_returns_silence_when_empty():
    cfg = AudioStreamConfig(chunk_size=8)
    ai = WebSocketAudioInput(cfg, queue.Queue(), poll_timeout=0.01)
    chunk = ai.capture_chunk()
    assert len(chunk) == 8 * 4
    assert np.count_nonzero(np.frombuffer(chunk, np.float32)) == 0


def test_output_enqueues_tagged_audio():
    q = queue.Queue()
    ao = WebSocketAudioOutput(AudioStreamConfig(), q)
    ao.play(b"pcmbytes")
    assert q.get_nowait() == ("audio", b"pcmbytes")
