from unittest.mock import MagicMock

import numpy as np

from app.rag.retriever import Retriever


def test_retriever_embeds_then_queries():
    fake_store = MagicMock()
    fake_store.query.return_value = [
        {"id": "arxiv_1_chunk_0", "document": "hello", "metadata": {"arxiv_id": "1"}, "distance": 0.1},
    ]
    fake_embedder = MagicMock()
    fake_embedder.embed_query.return_value = np.zeros(1024, dtype=np.float32)

    r = Retriever(embedder=fake_embedder, store=fake_store)
    results = r.retrieve("what is X", top_k=5)

    fake_embedder.embed_query.assert_called_once_with("what is X")
    fake_store.query.assert_called_once()
    assert results[0]["id"] == "arxiv_1_chunk_0"


def _chunk(paper_id: str, i: int, dist: float) -> dict:
    return {
        "id": f"arxiv_{paper_id}_chunk_{i}",
        "document": f"text {i}",
        "metadata": {"arxiv_id": paper_id},
        "distance": dist,
    }


def test_retriever_caps_chunks_per_paper():
    # Store returns 5 chunks, 4 of them from the same paper (the M1 symptom).
    candidates = [
        _chunk("A", 0, 0.10),
        _chunk("A", 1, 0.11),
        _chunk("A", 2, 0.12),
        _chunk("B", 0, 0.13),
        _chunk("A", 3, 0.14),
    ]
    fake_store = MagicMock()
    fake_store.query.return_value = candidates
    fake_embedder = MagicMock()
    fake_embedder.embed_query.return_value = np.zeros(1024, dtype=np.float32)

    r = Retriever(embedder=fake_embedder, store=fake_store)
    results = r.retrieve("q", top_k=5, max_per_paper=2)

    papers = [c["metadata"]["arxiv_id"] for c in results]
    assert papers.count("A") == 2  # capped at max_per_paper
    assert "B" in papers  # diversity preserved
    # Distance ordering kept: the two A chunks are the nearest two.
    assert [c["id"] for c in results if c["metadata"]["arxiv_id"] == "A"] == [
        "arxiv_A_chunk_0",
        "arxiv_A_chunk_1",
    ]


def test_retriever_over_fetches_candidates():
    # retrieve(top_k=5) should ask the store for more than 5 candidates so
    # dedup still has material to work with.
    fake_store = MagicMock()
    fake_store.query.return_value = []
    fake_embedder = MagicMock()
    fake_embedder.embed_query.return_value = np.zeros(1024, dtype=np.float32)

    Retriever(embedder=fake_embedder, store=fake_store).retrieve("q", top_k=5)

    _, kwargs = fake_store.query.call_args
    assert kwargs["top_k"] > 5
