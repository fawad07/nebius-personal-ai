#!/usr/bin/env python3
"""
scripts/nebius_spike.py — Phase 0 connectivity spike.

Proves three things before we wire the voice loop to Nebius:
  1. NEBIUS_API_KEY is set and accepted (auth works),
  2. the base_url endpoint is reachable,
  3. the chosen NVIDIA Nemotron model id is valid and returns a completion.

It deliberately depends on nothing but `openai` (no torch / TTS / project
imports), so you can run it the moment the key is in place — long before the
full environment installs.

Usage:
    export NEBIUS_API_KEY=...            # or put it in .env and `source` it
    python scripts/nebius_spike.py                      # uses config/settings.yaml model
    python scripts/nebius_spike.py --list               # list available model ids
    python scripts/nebius_spike.py --model nvidia/...    # override the model id
"""

from __future__ import annotations

import argparse
import os
import sys

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"


def _read_model_from_config() -> str:
    """Best-effort read of llm.model from config/settings.yaml; fall back to default."""
    try:
        import yaml  # optional; only if PyYAML is already installed
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "config", "settings.yaml")) as f:
            cfg = yaml.safe_load(f)
        return (cfg.get("llm", {}) or {}).get("model") or DEFAULT_MODEL
    except Exception:
        return DEFAULT_MODEL


def main() -> int:
    parser = argparse.ArgumentParser(description="Nebius Token Factory connectivity spike.")
    parser.add_argument("--model", default=None, help="Model id to test (default: config or built-in).")
    parser.add_argument("--list", action="store_true", help="List available model ids and exit.")
    parser.add_argument("--base-url", default=os.getenv("NEBIUS_BASE_URL", BASE_URL))
    args = parser.parse_args()

    api_key = os.getenv("NEBIUS_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: NEBIUS_API_KEY is not set. Get one at https://tokenfactory.nebius.com "
              "and `export NEBIUS_API_KEY=...` (or add it to .env and source it).", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: the `openai` package is not installed. Run:  pip install openai", file=sys.stderr)
        return 2

    client = OpenAI(base_url=args.base_url, api_key=api_key)

    if args.list:
        print(f"Models available at {args.base_url} :\n")
        try:
            for m in client.models.list().data:
                print("  ", m.id)
        except Exception as e:
            print(f"ERROR listing models: {e}", file=sys.stderr)
            return 1
        return 0

    model = args.model or _read_model_from_config()
    print(f"Endpoint : {args.base_url}")
    print(f"Model    : {model}")
    print("Sending a one-line test prompt...\n")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a terse assistant."},
                {"role": "user", "content": "In one sentence, confirm you are running and name yourself."},
            ],
            max_tokens=64,
        )
    except Exception as e:
        print(f"ERROR: the completion call failed: {e}\n", file=sys.stderr)
        print("Common causes: wrong model id (try --list), bad/expired key, or no credits.",
              file=sys.stderr)
        return 1

    reply = resp.choices[0].message.content
    print("REPLY:", reply)
    if resp.usage:
        print(f"\ntokens: prompt={resp.usage.prompt_tokens} "
              f"completion={resp.usage.completion_tokens} total={resp.usage.total_tokens}")
    print("\nOK — Nebius auth + endpoint + model all work. Phase 0 spike passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
