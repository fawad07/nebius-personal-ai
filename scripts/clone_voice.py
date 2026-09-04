"""
Voice cloning CLI.

Register a consented reference voice, then synthesize any text in that voice
and save a WAV recording.

    python -m scripts.clone_voice add   <name> <reference.wav>
    python -m scripts.clone_voice list
    python -m scripts.clone_voice say    <name> "text to speak" [--out file.wav]
    python -m scripts.clone_voice remove <name>

Cloning a voice is dual-use. `add` requires you to confirm you have the right
and consent to clone the voice in the reference clip. Generated audio is
synthetic and stays local.
"""

import argparse
import os
from datetime import datetime

from utils.config_loader import load_config
from src.tts.voice_cloner import VoiceCloner, VoiceLibrary


def _slug(text, limit=32):
    keep = "".join(c if c.isalnum() else "-" for c in text.lower())
    return "-".join(p for p in keep.split("-") if p)[:limit] or "clip"


def cmd_add(lib, name, ref):
    print("\n  Cloning a voice can be used to impersonate someone.")
    print("  Only register a voice you own or have explicit permission to clone.")
    ans = input(f"  Confirm you have consent to clone the voice in '{ref}'? (yes/no): ").strip().lower()
    if ans not in ("y", "yes"):
        print("Consent not given. Nothing was registered.")
        return
    note = input("  Optional note (whose voice / source): ").strip()
    lib.add(name, ref, consent=True, note=note)
    print(f"Registered voice '{name}'.")


def cmd_list(lib):
    names = lib.names()
    if not names:
        print("No registered voices. Add one with:  python -m scripts.clone_voice add <name> <ref.wav>")
        return
    print("Registered voices:")
    for n in names:
        print(f"  - {n}   ({lib.get(n)})")


def cmd_say(lib, cloner, recordings_path, name, text, out):
    ref = lib.get(name)
    if ref is None:
        print(f"Unknown voice '{name}'. Registered: {lib.names() or 'none'}")
        return
    if not out:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = os.path.join(recordings_path, f"{name}-{_slug(text)}-{ts}.wav")
    print(f"Synthesizing in '{name}' voice... (loads XTTS on first run)")
    cloner.clone_to_file(text, ref, out)
    print(f"Saved: {out}")


def cmd_remove(lib, name):
    print(f"Removed '{name}'." if lib.remove(name) else f"No voice named '{name}'.")


def main():
    ap = argparse.ArgumentParser(description="Clone a voice and synthesize recordings.")
    sub = ap.add_subparsers(dest="command", required=True)
    p_add = sub.add_parser("add"); p_add.add_argument("name"); p_add.add_argument("ref")
    sub.add_parser("list")
    p_say = sub.add_parser("say"); p_say.add_argument("name"); p_say.add_argument("text")
    p_say.add_argument("--out", default=None)
    p_rm = sub.add_parser("remove"); p_rm.add_argument("name")
    args = ap.parse_args()

    cfg = load_config("config/settings.yaml")
    vc = cfg.get("voice_clone", {})
    lib = VoiceLibrary(vc.get("voices_path", "data/voices"))
    recordings_path = vc.get("recordings_path", "data/recordings")

    if args.command == "add":
        cmd_add(lib, args.name, args.ref)
    elif args.command == "list":
        cmd_list(lib)
    elif args.command == "remove":
        cmd_remove(lib, args.name)
    elif args.command == "say":
        tts_cfg = cfg.get("tts", {})
        cloner = VoiceCloner.load(
            model_path=tts_cfg.get("model_path", "tts_models/multilingual/multi-dataset/xtts_v2")
        )
        cmd_say(lib, cloner, recordings_path, args.name, args.text, args.out)


if __name__ == "__main__":
    main()
