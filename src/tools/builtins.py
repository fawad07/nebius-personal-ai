import os
import re
from datetime import datetime

from src.tools.registry import Tool, ToolRegistry


def _get_current_time(args: dict) -> str:
    fmt = args.get("format") or "%A, %B %d, %I:%M %p"
    return datetime.now().astimezone().strftime(fmt)


def _make_remember(fact_store):
    def remember(args: dict) -> str:
        subject = str(args.get("subject", "")).strip() or "general"
        fact = str(args.get("fact", "")).strip()
        if not fact:
            return "(nothing to remember: missing 'fact')"
        fact_store.add(subject, fact)
        return f"Got it — I'll remember that ({fact})."
    return remember


def _make_recall(fact_store):
    def recall(args: dict) -> str:
        subject = str(args.get("subject", "")).strip() or None
        facts = fact_store.get(subject)
        if not facts:
            return "I don't have anything remembered about that yet."
        return "; ".join(facts)
    return recall


def _slug(text: str, limit: int = 32) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:limit] or "clip").strip("-")


def _make_synthesize_in_voice(cloner, library, recordings_path: str):
    def synthesize(args: dict) -> str:
        text = str(args.get("text", "")).strip()
        voice = str(args.get("voice", "")).strip()
        if not text:
            return "(nothing to synthesize: missing 'text')"
        if not voice:
            names = library.names()
            return f"(missing 'voice'. Registered voices: {names or 'none'})"
        ref = library.get(voice)
        if ref is None:
            return (
                f"(unknown voice '{voice}'. Registered voices: {library.names() or 'none'}. "
                "Register one first with scripts/clone_voice.py add)"
            )
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(recordings_path, f"{voice}-{_slug(text)}-{ts}.wav")
        try:
            cloner.clone_to_file(text, ref, out_path)
        except Exception as e:
            return f"(voice synthesis failed: {e})"
        return f"Saved a recording in the '{voice}' voice to {out_path}"
    return synthesize


def build_default_registry(fact_store=None, cloner=None, voice_library=None,
                           recordings_path: str = "data/recordings") -> ToolRegistry:
    """
    Assemble the built-in tool set. Memory tools are included only when a fact
    store is provided; the voice-cloning tool only when a cloner + voice
    library are provided (and it can only use already-registered, consented
    voices). All tools are local and side-effect-safe (no network).
    """
    tools = [
        Tool(
            name="get_current_time",
            description="Get the current local date and time.",
            params="()",
            func=_get_current_time,
        ),
    ]
    if fact_store is not None:
        tools.append(
            Tool(
                name="remember_fact",
                description=(
                    "Store a personal FACT or preference about the user — name, "
                    "favourites, likes/dislikes, details of their life. Use whenever "
                    'they say "remember that…", "my favourite … is …", "I like…". '
                    "This is the ONLY tool for personal facts (do NOT use save_note "
                    "for these)."
                ),
                params='{"subject": str, "fact": str}',
                func=_make_remember(fact_store),
            )
        )
        tools.append(
            Tool(
                name="recall_facts",
                description=(
                    "Retrieve facts stored with remember_fact. Use whenever the user "
                    'asks what you remember about them (e.g. "what\'s my favourite '
                    'colour?"). Always call this before saying you don\'t know.'
                ),
                params='{"subject": str (optional)}',
                func=_make_recall(fact_store),
            )
        )
    if cloner is not None and voice_library is not None:
        tools.append(
            Tool(
                name="synthesize_in_voice",
                description=(
                    "Speak the given text in one of the registered (consented) cloned "
                    "voices and save it as a WAV recording. Only works with voices that "
                    "have already been registered."
                ),
                params='{"text": str, "voice": <registered voice name>}',
                func=_make_synthesize_in_voice(cloner, voice_library, recordings_path),
            )
        )
    return ToolRegistry(tools)
