"""
src/tts/system_tts.py

Dependency-light, judge-proof TTS backed by the operating system's voice
(macOS `say`). A drop-in for VoiceCloningTTS: ``synthesize(text, mood)`` returns
float32 PCM bytes at ``sample_rate``.

Chosen for reliability over flash: it pulls in no torch / Coqui-TTS / numba /
llvmlite, downloads no models, and `say` emits 1-channel Float32 at the exact
target rate (verified: `say --data-format=LEF32@16000` → 16 kHz float mono), so
no resampling is needed. Mood maps to speaking rate. Unknown XTTS-only config
keys (model_path, voice_profile_path, ...) are accepted and ignored so the same
``tts:`` config block works for either engine.

The XTTS voice-cloning engine remains available (see voice_cloning_tts.py) for
when its install toolchain is in place; select it with ``tts.engine: xtts``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

logger = logging.getLogger("tts")

_BASE_WPM = 180  # base speaking rate for `say`; mood scales it


class SystemTTS:
    # Mood → speaking-rate multiplier (mirrors VoiceCloningTTS's prosody moods).
    _MOOD_RATE = {
        "neutral": 1.0,
        "calm": 0.92,
        "sad": 0.85,
        "excited": 1.15,
        "angry": 1.10,
        "stressed": 1.05,
    }

    def __init__(self, sample_rate: int = 16000, voice: str | None = None,
                 base_wpm: int = _BASE_WPM, **_ignored_xtts_keys):
        self.sample_rate = int(sample_rate)
        self.voice = voice
        self.base_wpm = int(base_wpm)
        self._say = shutil.which("say")
        if self._say is None:
            logger.warning(
                "System voice `say` not found (non-macOS host?). SystemTTS will "
                "return silent audio; the text reply still reaches the UI. For a "
                "non-macOS deploy, wire an espeak/piper backend here."
            )

    def _wpm_for_mood(self, mood: dict) -> int:
        factor = self._MOOD_RATE.get((mood or {}).get("mood", "neutral"), 1.0)
        return max(90, int(self.base_wpm * factor))

    def synthesize(self, text: str, mood: dict) -> bytes:
        """Render text → float32 PCM bytes at ``sample_rate`` via the OS voice.

        Never raises: any failure degrades to silence so a TTS hiccup can't
        crash a live conversation (the reply text still reaches the UI).
        """
        text = (text or "").strip()
        if not text or self._say is None:
            return b""
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name
            cmd = [
                self._say,
                "-r", str(self._wpm_for_mood(mood)),
                f"--data-format=LEF32@{self.sample_rate}",  # little-endian float32 @ target rate
                "-o", path,
            ]
            if self.voice:
                cmd += ["-v", self.voice]
            cmd += ["--", text]  # `--` guards text that begins with '-'
            subprocess.run(cmd, check=True, capture_output=True)

            audio, _sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:  # collapse to mono if a stereo voice ever appears
                audio = audio.mean(axis=1)
            return np.asarray(audio, dtype=np.float32).tobytes()
        except Exception:
            logger.exception("SystemTTS synthesis failed; returning silence.")
            return b""
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
