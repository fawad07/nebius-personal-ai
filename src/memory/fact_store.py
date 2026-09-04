import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone

logger = logging.getLogger("tools")


class FactStore:
    """
    Small SQLite store for personal facts the user asks the assistant to
    remember (e.g. "my daughter's name is Sara"), keyed by an optional subject.
    Persists across restarts; lives under data/ (gitignored, personal data).
    """

    def __init__(self, db_path: str = "data/facts.db"):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts      TEXT NOT NULL,
                    subject TEXT,
                    fact    TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def add(self, subject: str, fact: str):
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO facts (ts, subject, fact) VALUES (?, ?, ?)",
                    (datetime.now(timezone.utc).isoformat(), subject, fact),
                )
                self._conn.commit()
        except Exception:
            logger.exception("Failed to store fact.")

    def get(self, subject: str | None = None, limit: int = 20) -> list[str]:
        try:
            with self._lock:
                if subject:
                    rows = self._conn.execute(
                        "SELECT fact FROM facts WHERE subject = ? ORDER BY id DESC LIMIT ?",
                        (subject, limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT fact FROM facts ORDER BY id DESC LIMIT ?", (limit,)
                    ).fetchall()
        except Exception:
            logger.exception("Failed to read facts.")
            return []
        return [row["fact"] for row in rows]

    def close(self):
        with self._lock:
            self._conn.close()
