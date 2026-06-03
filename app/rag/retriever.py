from __future__ import annotations

from typing import Any

from app.stores.vector_store import VectorStore
from pipeline.embed import Embedder


class Retriever:
    """Embeds the query and looks up nearest chunks in the vector store.

    To avoid a single paper dominating the results (its adjacent chunks are
    near-duplicates and crowd out other papers), we over-fetch candidates and
    keep at most ``max_per_paper`` chunks per arxiv_id, preserving the
    distance ordering returned by the store.
    """

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(
        self, query: str, top_k: int = 5, max_per_paper: int = 2
    ) -> list[dict[str, Any]]:
        qvec = self.embedder.embed_query(query)
        # Over-fetch so dedup still yields top_k results even when the nearest
        # chunks are concentrated in a few papers.
        fetch_k = top_k * 5
        candidates = self.store.query(query_embedding=qvec.tolist(), top_k=fetch_k)
        return self._diversify(candidates, top_k=top_k, max_per_paper=max_per_paper)

    @staticmethod
    def _diversify(
        candidates: list[dict[str, Any]], top_k: int, max_per_paper: int
    ) -> list[dict[str, Any]]:
        """Keep candidates in distance order, capping chunks per arxiv_id."""
        seen: dict[str, int] = {}
        out: list[dict[str, Any]] = []
        for c in candidates:
            pid = c["metadata"].get("arxiv_id", "?")
            if seen.get(pid, 0) >= max_per_paper:
                continue
            seen[pid] = seen.get(pid, 0) + 1
            out.append(c)
            if len(out) >= top_k:
                break
        return out
