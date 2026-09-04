"""
automation/notes.py

Note-taking tools: save, list, and search the things the user asks the
assistant to remember — the "take a note" / "what were my notes" behaviour
a real assistant has.

Notes live in the encrypted memory DB (core/memory.py), same AES
protection as conversations and events — so they inherit the project's
encryption-at-rest guarantee rather than sitting in a second, plaintext
store. The stateless automation layer can't reach that DB, so the store is
injected at startup (run_assistant.build_agent), the same pattern as
reports._EVENT_SOURCE.
"""

from __future__ import annotations

import time

# Injected at startup: an object exposing store_note/list_notes/search_notes.
# None means note-taking isn't wired up (execute() then says so).
_STORE = None

MAX_LIST = 50


def _format(notes: list) -> str:
    if not notes:
        return "No notes yet."
    lines = []
    for n in notes:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(n["timestamp"]))
        lines.append(f"- [{when}] {n['content']}")
    return "\n".join(lines)


def execute(action: str, args: dict, grant=None):
    if _STORE is None:
        return "Note-taking is unavailable (no store configured)."

    if action == "save_note":
        content = str(args.get("content", "")).strip()
        if not content:
            return "There's nothing to note — tell me what to remember."
        try:
            _STORE.store_note(content)
        except Exception as e:
            return f"Couldn't save that note: {e}"
        return f"Noted: {content}"

    if action == "list_notes":
        try:
            limit = max(1, min(MAX_LIST, int(args.get("limit", 20))))
        except (TypeError, ValueError):
            limit = 20
        return _format(_STORE.list_notes(limit))

    if action == "search_notes":
        query = str(args.get("query", "")).strip()
        results = _STORE.search_notes(query)
        if not results:
            return f"No notes matching '{query}'." if query else "No notes yet."
        return _format(results)

    raise NotImplementedError(f"automation.notes: '{action}' not yet implemented")
