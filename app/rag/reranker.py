from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-encoder reranker (bge-reranker-base) over retrieved candidates."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        pairs = [(query, c["document"]) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]
