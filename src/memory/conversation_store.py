import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone

logger = logging.getLogger("conversation")


class ConversationStore:
    """
    SQLite-backed persistence for conversation turns, so history survives
    restarts and the agent can carry continuity across sessions instead of
    starting cold every launch.

    One row per turn: timestamp, session id, user text, reply text, and the
    mood dict (JSON). Thread-safe (the connection is shared across the async
    loop thread and worker threads), guarded by a lock.

    Note: transcripts are personal data. The DB lives under data/ (gitignored)
    and this store is opt-in via config; see privacy notes in the README.
    """

    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts         TEXT NOT NULL,
                    session_id TEXT,
                    user_text  TEXT,
                    reply_text TEXT,
                    mood       TEXT
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_id ON turns (id)")
            self._conn.commit()

    def append(self, session_id: str, user_text: str, reply_text: str, mood: dict | None):
        """Persist one completed turn."""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO turns (ts, session_id, user_text, reply_text, mood) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        session_id,
                        user_text,
                        reply_text,
                        json.dumps(mood or {}),
                    ),
                )
                self._conn.commit()
        except Exception:  # persistence must never break the conversation
            logger.exception("Failed to persist conversation turn.")

    def recent(self, limit: int = 20) -> list[dict]:
        """
        Return up to ``limit`` most recent turns in chronological (oldest ->
        newest) order, shaped like ConversationManager's in-memory history:
        {"user", "reply", "mood"}.
        """
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT user_text, reply_text, mood FROM turns ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        except Exception:
            logger.exception("Failed to load conversation history.")
            return []

        history = []
        for row in reversed(rows):  # DESC fetch -> reverse to chronological
            try:
                mood = json.loads(row["mood"]) if row["mood"] else {}
            except (json.JSONDecodeError, TypeError):
                mood = {}
            history.append({"user": row["user_text"], "reply": row["reply_text"], "mood": mood})
        return history

    def clear(self):
        with self._lock:
            self._conn.execute("DELETE FROM turns")
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()
