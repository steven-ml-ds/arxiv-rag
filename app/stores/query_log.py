from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any

_COLUMNS = (
    "ts", "question", "rewritten", "doc_ids", "latency_ms", "input_tokens", "output_tokens",
)


class QueryLog:
    """SQLite log of chat requests: latency, retrieved doc ids, token usage.

    Process-wide singleton queried from FastAPI's threadpool, so the connection
    is opened with check_same_thread=False and serialised by a lock (same
    pattern as KeywordStore).
    """

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS queries ("
                "ts REAL, question TEXT, rewritten TEXT, doc_ids TEXT, "
                "latency_ms REAL, input_tokens INTEGER, output_tokens INTEGER)"
            )
            self.conn.commit()

    def log(
        self,
        question: str,
        rewritten: str,
        doc_ids: list[str],
        latency_ms: float,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO queries VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), question, rewritten, ",".join(doc_ids),
                 latency_ms, input_tokens, output_tokens),
            )
            self.conn.commit()

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT ts, question, rewritten, doc_ids, latency_ms, input_tokens, "
                "output_tokens FROM queries ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(zip(_COLUMNS, r)) for r in rows]
