from __future__ import annotations

from app.stores.vector_store import VectorStore
from pipeline.chunk import Chunk
from pipeline.fetch_arxiv import Paper


def build_records(
    paper: Paper, chunks: list[Chunk]
) -> tuple[list[str], list[str], list[dict]]:
    """Stable ids + documents + JSON-safe metadata for a paper's chunks."""
    ids = [f"arxiv_{paper.arxiv_id}_chunk_{i}" for i in range(len(chunks))]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": ", ".join(paper.authors),
            "year": paper.published_year,
            "chunk_index": i,
            "section": chunks[i].section,
        }
        for i in range(len(chunks))
    ]
    return ids, documents, metadatas


def index_paper(
    paper: Paper,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    store: VectorStore,
    keyword_store=None,
) -> None:
    """Write a paper's chunks into the vector store (and keyword store if given).

    Chunk IDs are stable: arxiv_<id>_chunk_<i> so re-runs replace correctly.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
        )
    ids, documents, metadatas = build_records(paper, chunks)
    store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    if keyword_store is not None:
        keyword_store.add(ids=ids, documents=documents)
