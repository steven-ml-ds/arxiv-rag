from __future__ import annotations

from typing import Any

from app.rag.fusion import reciprocal_rank_fusion
from app.stores.vector_store import VectorStore
from pipeline.embed import Embedder


class Retriever:
    """Hybrid retrieval: dense (Chroma) + sparse (FTS5 BM25), fused with RRF.

    Over-fetches `candidate_k` from each source, fuses the rankings, materialises
    full records (reusing vector hits, fetching keyword-only ids via store.get),
    optionally reranks with a cross-encoder, then caps chunks per paper so one
    paper can't dominate.
    """

    def __init__(self, embedder: Embedder, store: VectorStore, keyword_store, reranker=None):
        self.embedder = embedder
        self.store = store
        self.keyword_store = keyword_store
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        max_per_paper: int = 2,
    ) -> list[dict[str, Any]]:
        qvec = self.embedder.embed_query(query)
        vec_hits = self.store.query(query_embedding=qvec.tolist(), top_k=candidate_k)
        kw_hits = self.keyword_store.query(query, top_k=candidate_k)

        by_id: dict[str, dict[str, Any]] = {h["id"]: h for h in vec_hits}
        fused_ids = reciprocal_rank_fusion(
            [[h["id"] for h in vec_hits], [h["id"] for h in kw_hits]]
        )

        missing = [i for i in fused_ids if i not in by_id]
        if missing:
            for rec in self.store.get(missing):
                by_id[rec["id"]] = rec

        ordered = [by_id[i] for i in fused_ids if i in by_id]
        if self.reranker is not None:
            ordered = self.reranker.rerank(query, ordered, top_k=candidate_k)
        return self._diversify(ordered, top_k=top_k, max_per_paper=max_per_paper)

    @staticmethod
    def _diversify(
        candidates: list[dict[str, Any]], top_k: int, max_per_paper: int
    ) -> list[dict[str, Any]]:
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
