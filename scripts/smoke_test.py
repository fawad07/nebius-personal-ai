"""
Offline pipeline smoke test.

Loads the real on-device models (VAD, STT, speaker embedding, SER) and runs a
synthetic audio buffer through them -- no microphone, no OpenAI key, no TTS
reference voice required. Proves the local perception stack actually loads and
executes end to end.

Usage:
    python -m scripts.smoke_test [--stt-model tiny]
"""

import os

# Must be set before torch / CTranslate2 import (see src/main.py for why).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
import time

import numpy as np

from src.audio.vad import VoiceActivityDetector
from src.ser.emotion_classifier import EmotionClassifier
from src.speaker.embedding_model import SpeakerEmbeddingModel
from src.speaker.speaker_database import SpeakerDatabase
from src.speaker.speaker_identifier import SpeakerIdentifier
from src.stt.stt_service import STTService

SR = 16000


def make_tone(seconds=1.0, freq=180.0, sr=SR):
    """A voiced-ish tone with an amplitude envelope (stand-in for speech)."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    env = np.hanning(t.size).astype(np.float32)
    sig = 0.2 * np.sin(2 * np.pi * freq * t) + 0.05 * np.sin(2 * np.pi * 2 * freq * t)
    return (sig.astype(np.float32) * env)


def step(name, fn):
    t0 = time.time()
    try:
        out = fn()
        print(f"  [OK]   {name:22s} ({time.time()-t0:5.1f}s) -> {out}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name:22s} ({time.time()-t0:5.1f}s) -> {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt-model", default="tiny", help="faster-whisper model (tiny keeps the test fast)")
    args = ap.parse_args()

    audio = make_tone()
    audio_bytes = audio.tobytes()
    silence_bytes = np.zeros(SR // 2, dtype=np.float32).tobytes()
    print(f"Synthetic audio: {audio.size} samples @ {SR} Hz ({audio.nbytes} bytes)\n")

    ok = []

    print("VAD (Silero):")
    vad = VoiceActivityDetector(sample_rate=SR)
    ok.append(step("is_speech(tone)", lambda: vad.is_speech(audio_bytes)))
    ok.append(step("is_speech(silence)", lambda: vad.is_speech(silence_bytes)))

    print("\nSER (acoustic heuristic):")
    ser = EmotionClassifier(sample_rate=SR)
    ok.append(step("classify(tone)", lambda: ser.classify(audio_bytes)))

    print(f"\nSTT (faster-whisper '{args.stt_model}'):")
    stt = STTService(model_path=args.stt_model, use_gpu=False)
    ok.append(step("transcribe(tone)", lambda: repr(stt.transcribe(audio_bytes))))

    print("\nSpeaker (ECAPA-TDNN + identify):")
    embed = SpeakerEmbeddingModel()
    db = SpeakerDatabase()
    ident = SpeakerIdentifier(embed, db, threshold=0.75)
    ok.append(step("embed(tone).shape", lambda: embed.embed(audio_bytes).shape))
    ok.append(step("identify_detailed", lambda: ident.identify_detailed(audio_bytes).group))

    passed, total = sum(ok), len(ok)
    print(f"\n=== {passed}/{total} pipeline stages OK ===")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
