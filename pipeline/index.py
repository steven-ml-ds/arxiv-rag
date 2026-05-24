from __future__ import annotations

from app.stores.vector_store import VectorStore
from pipeline.fetch_arxiv import Paper


def index_paper(
    paper: Paper,
    chunks: list[str],
    embeddings: list[list[float]],
    store: VectorStore,
) -> None:
    """Write a paper's chunks + embeddings + metadata into the vector store.

    Chunk IDs are stable: arxiv_<id>_chunk_<i> so re-runs replace correctly.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
        )

    ids = [f"arxiv_{paper.arxiv_id}_chunk_{i}" for i in range(len(chunks))]
    # Chroma requires JSON-safe metadata values — coerce author list to string.
    metadatas = [
        {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": ", ".join(paper.authors),
            "year": paper.published_year,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]
    store.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
