from __future__ import annotations

import re
import sqlite3
import threading
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _to_match_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 MATCH expression (terms OR'd)."""
    terms = _TOKEN_RE.findall(text.lower())
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


class KeywordStore:
    """SQLite FTS5 BM25 keyword index over chunk documents.

    Mirrors VectorStore's role for sparse retrieval. bm25() returns lower =
    better, so we negate it into a descending 'score' for a uniform interface.

    The store is a process-wide singleton (see app.api.deps) but is queried from
    FastAPI's threadpool workers, so the connection is opened with
    ``check_same_thread=False`` and every DB access is serialised by a lock
    (sqlite has no row-level concurrency for a shared connection).
    """

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
                "USING fts5(chunk_id UNINDEXED, document)"
            )
            self.conn.commit()

    def add(self, ids: list[str], documents: list[str]) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.executemany("DELETE FROM chunks WHERE chunk_id = ?", [(i,) for i in ids])
            cur.executemany(
                "INSERT INTO chunks(chunk_id, document) VALUES (?, ?)",
                list(zip(ids, documents)),
            )
            self.conn.commit()

    def query(self, text: str, top_k: int = 20) -> list[dict[str, Any]]:
        match = _to_match_query(text)
        if match is None:
            return []
        with self._lock:
            rows = self.conn.execute(
                "SELECT chunk_id, bm25(chunks) AS score FROM chunks "
                "WHERE chunks MATCH ? ORDER BY score LIMIT ?",
                (match, top_k),
            ).fetchall()
        # bm25 lower=better -> negate so larger score = more relevant
        return [{"id": r[0], "score": -float(r[1])} for r in rows]
