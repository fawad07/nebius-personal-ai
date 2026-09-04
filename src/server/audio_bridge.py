"""
Audio adapters that make a browser (over WebSocket) act as the microphone and
speaker, so the *entire* existing pipeline (SessionManager, VAD, STT, ...) runs
unchanged -- only the audio source/sink are swapped.

These conform to the same interface as src/audio/input.py and output.py:
``.config`` (an AudioStreamConfig), ``start()``, ``stop()``, plus
``capture_chunk() -> bytes`` (input) and ``play(bytes)`` (output).

Bridging uses plain thread-safe queues: SessionManager runs its blocking audio
calls in worker threads (via asyncio.to_thread), while the WebSocket server
runs on its own loop, so a thread-safe hand-off is exactly what's needed.
"""

import queue

import numpy as np

_FLOAT32_BYTES = 4


class WebSocketAudioInput:
    """
    Mic replacement fed by audio frames arriving from the browser.

    Incoming bytes are buffered and re-chunked to exactly ``config.chunk_size``
    float32 samples, so browser frame sizes don't have to match the pipeline's
    chunk size. When no audio is available, ``capture_chunk`` returns a chunk of
    silence, which VAD reads as non-speech -- the loop simply idles.
    """

    def __init__(self, config, in_queue: queue.Queue, poll_timeout: float = 0.1):
        self.config = config
        self._q = in_queue
        self._poll_timeout = poll_timeout
        self._buf = bytearray()
        self._chunk_bytes = config.chunk_size * _FLOAT32_BYTES
        self._silence = np.zeros(config.chunk_size, dtype=np.float32).tobytes()
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def capture_chunk(self) -> bytes:
        # Fill the buffer until we have a full chunk, or time out into silence.
        while len(self._buf) < self._chunk_bytes:
            try:
                self._buf += self._q.get(timeout=self._poll_timeout)
            except queue.Empty:
                if len(self._buf) == 0:
                    return self._silence
                # Pad a partial tail with silence so we still emit a full chunk.
                self._buf += self._silence[: self._chunk_bytes - len(self._buf)]
                break
        chunk = bytes(self._buf[: self._chunk_bytes])
        del self._buf[: self._chunk_bytes]
        return chunk


class WebSocketAudioOutput:
    """
    Speaker replacement: reply audio is pushed to an outbound queue tagged
    ``("audio", bytes)`` for the server to stream to the browser.
    """

    def __init__(self, config, out_queue: queue.Queue):
        self.config = config
        self._out = out_queue

    def start(self):
        pass

    def stop(self):
        pass

    def play(self, audio_bytes: bytes):
        self._out.put(("audio", audio_bytes))
