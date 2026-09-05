"""
src/memory/notes_store.py

A small SQLite-backed store for the note-taking automation tools
(automation/notes.py: save_note / list_notes / search_notes).

In the original offline_assistant these notes lived in its encrypted SQLCipher
memory DB, injected at startup. This merged project uses plain SQLite for its
conversation/fact memory (see the project decision to defer encryption until the
face-auth stretch goal), so notes get a matching plain-SQLite store here. It
exposes exactly the interface automation.notes expects:
``store_note(content)`` / ``list_notes(limit)`` / ``search_notes(query)``,
each note a dict of ``{"timestamp": float, "content": str}``.
"""

from __future__ import annotations

import os
import sqlite3
import time


class SQLiteNotesStore:
    def __init__(self, db_path: str = "data/notes.db"):
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._db_path = db_path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init(self) -> None:
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS notes ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "timestamp REAL NOT NULL, "
                "content TEXT NOT NULL)"
            )

    def store_note(self, content: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO notes (timestamp, content) VALUES (?, ?)",
                (time.time(), content),
            )

    def list_notes(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT timestamp, content FROM notes ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [{"timestamp": r[0], "content": r[1]} for r in rows]

    def search_notes(self, query: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT timestamp, content FROM notes WHERE content LIKE ? ORDER BY id DESC",
                (f"%{query}%",),
            ).fetchall()
        return [{"timestamp": r[0], "content": r[1]} for r in rows]
