from __future__ import annotations

from typing import Any

from app.stores.vector_store import VectorStore
from pipeline.embed import Embedder


class Retriever:
    """Embeds the query and looks up nearest chunks in the vector store."""

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        qvec = self.embedder.embed_query(query)
        return self.store.query(query_embedding=qvec.tolist(), top_k=top_k)
