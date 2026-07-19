from __future__ import annotations

from typing import Any

import chromadb


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection.

    Owns the client and exposes only add/query/count. Replacing Chroma with
    another vector DB later means rewriting this file alone.
    """

    def __init__(self, path: str, collection: str):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        return [
            {
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            }
            for i in range(len(res["ids"][0]))
        ]

    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        res = self.collection.get(ids=ids, include=["documents", "metadatas"])
        return [
            {
                "id": res["ids"][i],
                "document": res["documents"][i],
                "metadata": res["metadatas"][i],
                "distance": None,
            }
            for i in range(len(res["ids"]))
        ]

    def scan(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Page through the collection without a query (for sampling/exports)."""
        res = self.collection.get(
            limit=limit, offset=offset, include=["documents", "metadatas"]
        )
        return [
            {
                "id": res["ids"][i],
                "document": res["documents"][i],
                "metadata": res["metadatas"][i],
                "distance": None,
            }
            for i in range(len(res["ids"]))
        ]

    def count(self) -> int:
        return self.collection.count()
