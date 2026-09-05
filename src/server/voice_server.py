"""
Browser voice client server.

Serves a static web page and a WebSocket endpoint. The browser captures mic
audio and streams 16 kHz float32 frames as binary WS messages; the server feeds
them into the real voice pipeline via the WebSocket audio bridge (so nothing in
SessionManager/VAD/STT/... changes), and streams back live turn state, partial
transcripts, the reply text, and the reply audio for playback.

Modes:
  --mock  (default when models/key aren't set up) drives the UI end to end with
          a scripted response and a beep -- no models, key, or mic-side ML
          required, so the page/protocol/state view can be exercised anywhere.
  --real  builds the full Agent pipeline with the browser as mic + speaker.
          Needs the models, an LLM key (or local provider), and a voice sample.

Run:  python -m src.server.voice_server            # mock
      python -m src.server.voice_server --real
"""

import os

# torch and CTranslate2 (faster-whisper's backend) each bundle their own
# libiomp5.dylib on macOS; loading both aborts with "OMP: Error #15" unless
# duplicate OpenMP runtimes are allowed, and pinning to one thread keeps them
# from segfaulting. src/main.py sets these for the CLI; this server is a
# separate entry point, so it must set them too — before torch is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import asyncio
import functools
import logging
import math
import queue
import struct
import threading
import time

import websockets
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from src.audio.stream_config import AudioStreamConfig
from src.server.audio_bridge import WebSocketAudioInput, WebSocketAudioOutput
from src.server.protocol import decode, encode

logger = logging.getLogger("voice_server")

_WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "web")


def _tone(seconds: float, freq: float = 440.0, rate: int = 16000) -> bytes:
    """A short float32 beep, so mock mode has something to play back."""
    n = int(seconds * rate)
    env = [min(1.0, min(i, n - i) / (0.02 * rate)) for i in range(n)]  # fade in/out
    samples = (0.2 * env[i] * math.sin(2 * math.pi * freq * i / rate) for i in range(n))
    return b"".join(struct.pack("<f", s) for s in samples)


# --------------------------------------------------------------------------- #
# Per-connection session: two queues bridge the WS loop and the runner thread.
# --------------------------------------------------------------------------- #
class Connection:
    def __init__(self):
        self.inbound: queue.Queue = queue.Queue()    # ("audio", bytes) | ("text", str) | ("stop", None)
        self.outbound: queue.Queue = queue.Queue()   # ("json", str) | ("audio", bytes)

    def send_json(self, msg_type: str, **fields):
        self.outbound.put(("json", encode(msg_type, **fields)))

    def send_audio(self, data: bytes):
        self.outbound.put(("audio", data))


class MockRunner:
    """Scripted responder: no models needed, exercises the whole UI + protocol."""

    def __init__(self, conn: Connection, rate: int = 16000):
        self.conn = conn
        self.rate = rate

    def run(self):
        self.conn.send_json("state", state="listening")
        while True:
            kind, payload = self.conn.inbound.get()
            if kind == "stop":
                break
            if kind == "text":
                self._respond(payload)
            # In mock mode raw audio isn't transcribed; nudge the indicator only.
            elif kind == "audio":
                self.conn.send_json("meta", hearing=True)

    def _respond(self, text: str):
        self.conn.send_json("transcript", text=text, final=True)
        self.conn.send_json("state", state="thinking")
        time.sleep(0.4)
        self.conn.send_json("meta", speaker="self", emotion="calm")
        reply = f"(mock) You said: {text}"
        self.conn.send_json("state", state="speaking")
        self.conn.send_json("reply", text=reply, mood="neutral")
        self.conn.send_audio(_tone(0.5, rate=self.rate))
        time.sleep(0.5)
        self.conn.send_json("state", state="listening")


class RealRunner:
    """Runs the full Agent pipeline with the browser as microphone + speaker."""

    def __init__(self, conn: Connection, config_path: str = "config/settings.yaml"):
        self.conn = conn
        self.config_path = config_path
        self._audio_in_queue: queue.Queue = queue.Queue()

    def _on_state(self, old, new, ms):
        self.conn.send_json("state", state=new.value)

    def _on_partial(self, text):
        self.conn.send_json("transcript", text=text, final=False)

    def _on_reply(self, user_text, reply_text, mood):
        self.conn.send_json("transcript", text=user_text, final=True)
        self.conn.send_json("reply", text=reply_text, mood=mood.get("mood", "neutral"))
        self.conn.send_json("meta", **mood)

    def run(self):
        from src.agent.agent import Agent  # deferred: heavy imports (torch/TTS)

        cfg = AudioStreamConfig()  # rate/chunk come from the Agent's own config
        audio_in = WebSocketAudioInput(cfg, self._audio_in_queue)
        audio_out = WebSocketAudioOutput(cfg, self.conn.outbound)

        agent = Agent(
            config_path=self.config_path,
            audio_in=audio_in,
            audio_out=audio_out,
            on_state_change=self._on_state,
            on_partial_transcript=self._on_partial,
            on_reply=self._on_reply,
            # Force always-on VAD: push-to-talk reads stdin (EOF in a server),
            # which would end the session instantly and close the stores.
            input_mode="vad",
        )

        # Feed inbound browser audio into the mic bridge; handle typed text as a
        # full turn (skips STT); stop on disconnect.
        def pump():
            while True:
                kind, payload = self.conn.inbound.get()
                if kind == "stop":
                    agent.session_manager.stop()
                    break
                if kind == "audio":
                    self._audio_in_queue.put(payload)
                elif kind == "text":
                    self._handle_text(agent, payload)
        threading.Thread(target=pump, daemon=True).start()

        self.conn.send_json("info", message="pipeline ready")
        agent.run()

    def _handle_text(self, agent, text: str):
        """Run a typed message as a full turn (no mic/STT): reason with
        Nemotron + tools, echo the exchange to the UI, and speak the reply.

        Lets a judge test the whole pipeline by typing — no microphone, no room
        noise — which is why the browser demo is the reliable demo surface.
        """
        text = (text or "").strip()
        if not text:
            return
        try:
            self.conn.send_json("transcript", text=text, final=True)
            result = agent.conversation_manager.respond_with_mood(
                text, {"label": "neutral"}, "primary"
            )
            reply = result.get("reply", "")
            self.conn.send_json("reply", text=reply, mood=result.get("mood"))
            # Persist the exchange so memory carries across turns/sessions.
            try:
                mood = {
                    "mood": result.get("mood"),
                    "intent": result.get("intent"),
                    "sensitivity": result.get("sensitivity"),
                }
                agent.conversation_manager.update_history(text, reply, mood)
            except Exception:
                pass
            # Speak the reply back to the browser (best-effort).
            try:
                audio = agent.tts.synthesize(reply, {"mood": result.get("mood", "neutral")})
                if audio:
                    self.conn.send_audio(audio)
            except Exception:
                logger.exception("TTS for typed turn failed (reply still sent).")
        except Exception:
            logger.exception("Text turn failed.")
            self.conn.send_json("reply", text="(sorry — that turn failed)")


# --------------------------------------------------------------------------- #
# WebSocket handling
# --------------------------------------------------------------------------- #
async def _ws_handler(ws, mock: bool):
    conn = Connection()
    runner = MockRunner(conn) if mock else RealRunner(conn)
    runner_thread = threading.Thread(target=runner.run, daemon=True)
    runner_thread.start()
    logger.info("Client connected (%s mode).", "mock" if mock else "real")

    async def sender():
        while True:
            kind, payload = await asyncio.to_thread(conn.outbound.get)
            if kind == "json":
                await ws.send(payload)
            elif kind == "audio":
                await ws.send(payload)

    send_task = asyncio.create_task(sender())
    try:
        async for message in ws:
            if isinstance(message, bytes):
                conn.inbound.put(("audio", message))
            else:
                try:
                    msg = decode(message)
                except ValueError:
                    continue
                if msg["type"] == "text" and msg.get("text"):
                    conn.inbound.put(("text", msg["text"]))
    except websockets.ConnectionClosed:
        pass
    finally:
        conn.inbound.put(("stop", None))
        send_task.cancel()
        logger.info("Client disconnected.")


def _serve_static(host: str, port: int):
    handler = functools.partial(SimpleHTTPRequestHandler, directory=os.path.abspath(_WEB_DIR))
    httpd = ThreadingHTTPServer((host, port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def _main_async(host: str, http_port: int, ws_port: int, mock: bool):
    _serve_static(host, http_port)
    handler = functools.partial(_ws_handler, mock=mock)
    async with websockets.serve(handler, host, ws_port, max_size=None):
        logger.info("Open  http://%s:%d/?ws=%d   (%s mode)", host, http_port, ws_port,
                    "MOCK" if mock else "REAL")
        await asyncio.Future()  # run forever


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Browser voice client server")
    ap.add_argument("--real", action="store_true", help="run the full pipeline (needs models/key/voice sample)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--ws-port", type=int, default=8765)
    args = ap.parse_args()
    try:
        asyncio.run(_main_async(args.host, args.http_port, args.ws_port, mock=not args.real))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
