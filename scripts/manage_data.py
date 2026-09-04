"""
Personal-data management: see what the agent has stored about you, and delete
it. Withdrawing consent means being able to erase the data, so this is part of
the consent story.

Usage:
    python -m scripts.manage_data list
    python -m scripts.manage_data forget <speaker-name>
    python -m scripts.manage_data wipe-memory          # conversation + facts
"""

import argparse
import json
import os

from utils.config_loader import load_config


def _load(path, default):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return default


def cmd_list(cfg):
    profiles_path = cfg["speaker"]["profiles_path"]
    groups = _load(os.path.join(profiles_path, "groups.json"), {})
    consent = _load(os.path.join(profiles_path, "consent.json"), {})

    print("Enrolled speakers (voiceprints):")
    if not groups:
        print("  (none)")
    for name, group in sorted(groups.items()):
        c = consent.get(name, {})
        mark = "consent ✓" if c.get("consented") else "no consent record"
        enc = "encrypted" if os.path.exists(os.path.join(profiles_path, f"{name}.npy.enc")) else "plaintext"
        print(f"  - {name:16s} group={group:8s} {enc:9s} {mark}")

    for label, path in (("conversation memory", cfg.get("memory", {}).get("db_path", "data/memory.db")),
                        ("remembered facts", cfg.get("tools", {}).get("facts_db_path", "data/facts.db"))):
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        print(f"{label}: {'present' if size else 'none'} ({path})")


def cmd_forget(cfg, name):
    profiles_path = cfg["speaker"]["profiles_path"]
    removed = []
    for ext in (".npy", ".npy.enc"):
        p = os.path.join(profiles_path, f"{name}{ext}")
        if os.path.exists(p):
            os.remove(p)
            removed.append(p)

    for meta in ("groups.json", "consent.json"):
        p = os.path.join(profiles_path, meta)
        data = _load(p, {})
        if name in data:
            del data[name]
            with open(p, "w") as f:
                json.dump(data, f, indent=2)
            removed.append(f"{p}#{name}")

    if removed:
        print(f"Forgot '{name}'. Removed: {', '.join(removed)}")
    else:
        print(f"No stored data found for '{name}'.")


def cmd_wipe_memory(cfg):
    targets = [
        cfg.get("memory", {}).get("db_path", "data/memory.db"),
        cfg.get("tools", {}).get("facts_db_path", "data/facts.db"),
    ]
    for p in targets:
        if os.path.isfile(p):
            os.remove(p)
            print(f"Deleted {p}")
        else:
            print(f"(nothing at {p})")


def main():
    ap = argparse.ArgumentParser(description="Manage the voice agent's stored personal data.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show stored speakers and memory")
    p_forget = sub.add_parser("forget", help="delete one speaker's voiceprint + records")
    p_forget.add_argument("name")
    sub.add_parser("wipe-memory", help="delete conversation history and remembered facts")

    args = ap.parse_args()
    cfg = load_config("config/settings.yaml")

    if args.command == "list":
        cmd_list(cfg)
    elif args.command == "forget":
        cmd_forget(cfg, args.name)
    elif args.command == "wipe-memory":
        cmd_wipe_memory(cfg)


if __name__ == "__main__":
    main()
