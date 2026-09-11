#!/usr/bin/env python3
"""
scripts/intro.py — the assistant introduces itself, out loud, in its own voice.

A ready-made opening shot for the demo video: it speaks a short self-introduction
using the same voice engine the assistant uses (macOS `say`). By default it
speaks a curated intro; with --dynamic it asks the Nemotron model on Nebius to
write the introduction from its own real tool list (needs NEBIUS_API_KEY).

Examples:
    python scripts/intro.py                      # speak the default intro aloud
    python scripts/intro.py --print              # just print the text
    python scripts/intro.py --save intro.wav     # save a clean WAV for editing
    python scripts/intro.py --dynamic            # let Nemotron write it, then speak
    python scripts/intro.py --voice Samantha     # pick a macOS voice (say -v '?')
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_INTRO = (
    "Hi — I'm your personal AI assistant. I run privately on your own machine and "
    "think with an NVIDIA Nemotron model on Nebius. I can hear how you sound, "
    "remember what you tell me across our conversations, take and search your notes, "
    "and look things up on the web with Tavily. My speech recognition, speaker "
    "recognition, and emotion sensing all run on your device — only my reasoning "
    "leaves, to an open model you could host yourself. Ask me something, or just "
    "start talking."
)


def generate_intro() -> str:
    """Ask the Nemotron model to introduce itself from its real tool list."""
    sys.path.insert(0, _ROOT)
    from webapp.agent_core import AgentCore  # torch-free core

    core = AgentCore()
    tools = ", ".join(core.tool_names())
    system = "You are a friendly personal AI assistant introducing yourself."
    user = (
        "Introduce yourself in 4-5 spoken sentences for a demo video. First person, "
        "warm, concise, natural to say out loud (no markdown, no lists). Mention that "
        "you reason with an NVIDIA Nemotron model on Nebius, that the user's data stays "
        "private and on-device, and weave in these abilities: " + tools + "."
    )
    return core.llm.generate_text(system, user).strip()


def speak(text: str, voice: str | None, rate: int, save: str | None) -> None:
    if save:
        # Render a clean WAV via the project's own system-voice backend.
        sys.path.insert(0, _ROOT)
        import numpy as np
        import soundfile as sf
        from src.tts.system_tts import SystemTTS

        tts = SystemTTS(voice=voice, base_wpm=rate)
        audio = tts.synthesize(text, {"mood": "neutral"})
        arr = np.frombuffer(audio, dtype=np.float32)
        sf.write(save, arr, tts.sample_rate)
        print(f"Saved {len(arr) / tts.sample_rate:.1f}s of audio to {save}")
        return

    cmd = ["say", "-r", str(rate)]
    if voice:
        cmd += ["-v", voice]
    cmd += ["--", text]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Speak the assistant's self-introduction.")
    ap.add_argument("--dynamic", action="store_true",
                    help="Have the Nemotron model write the intro (needs NEBIUS_API_KEY).")
    ap.add_argument("--voice", default=None, help="macOS voice name (list: say -v '?').")
    ap.add_argument("--rate", type=int, default=180, help="Speaking rate (words/min).")
    ap.add_argument("--save", default=None, help="Save a WAV file instead of speaking aloud.")
    ap.add_argument("--print", dest="show", action="store_true", help="Print the text and exit.")
    args = ap.parse_args()

    try:
        text = generate_intro() if args.dynamic else DEFAULT_INTRO
    except Exception as e:
        print(f"(dynamic intro failed: {e}\n falling back to the default intro)", file=sys.stderr)
        text = DEFAULT_INTRO

    if args.show:
        print(text)
        return 0

    speak(text, voice=args.voice, rate=args.rate, save=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
