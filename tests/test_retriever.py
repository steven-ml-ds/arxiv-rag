from unittest.mock import MagicMock

import numpy as np

from app.rag.retriever import Retriever


def _embedder():
    e = MagicMock()
    e.embed_query.return_value = np.zeros(1024, dtype=np.float32)
    return e


def test_retriever_embeds_then_queries():
    fake_store = MagicMock()
    fake_store.query.return_value = [
        {"id": "arxiv_1_chunk_0", "document": "hello", "metadata": {"arxiv_id": "1"}, "distance": 0.1},
    ]
    fake_kw = MagicMock()
    fake_kw.query.return_value = []
    fake_embedder = _embedder()

    r = Retriever(embedder=fake_embedder, store=fake_store, keyword_store=fake_kw)
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
    candidates = [
        _chunk("A", 0, 0.10),
        _chunk("A", 1, 0.11),
        _chunk("A", 2, 0.12),
        _chunk("B", 0, 0.13),
        _chunk("A", 3, 0.14),
    ]
    fake_store = MagicMock()
    fake_store.query.return_value = candidates
    fake_kw = MagicMock()
    fake_kw.query.return_value = []
    r = Retriever(embedder=_embedder(), store=fake_store, keyword_store=fake_kw)
    results = r.retrieve("q", top_k=5, max_per_paper=2)

    papers = [c["metadata"]["arxiv_id"] for c in results]
    assert papers.count("A") == 2
    assert "B" in papers
    assert [c["id"] for c in results if c["metadata"]["arxiv_id"] == "A"] == [
        "arxiv_A_chunk_0",
        "arxiv_A_chunk_1",
    ]


def test_retriever_over_fetches_candidates():
    fake_store = MagicMock()
    fake_store.query.return_value = []
    fake_kw = MagicMock()
    fake_kw.query.return_value = []
    Retriever(embedder=_embedder(), store=fake_store, keyword_store=fake_kw).retrieve("q", top_k=5)

    _, kwargs = fake_store.query.call_args
    assert kwargs["top_k"] > 5


def test_retriever_fuses_vector_and_keyword():
    fake_store = MagicMock()
    fake_store.query.return_value = [
        {"id": "A_0", "document": "x", "metadata": {"arxiv_id": "A"}, "distance": 0.1},
    ]
    fake_store.get.return_value = [
        {"id": "B_0", "document": "y", "metadata": {"arxiv_id": "B"}, "distance": None},
    ]
    fake_kw = MagicMock()
    fake_kw.query.return_value = [{"id": "B_0", "score": 9.0}]

    r = Retriever(embedder=_embedder(), store=fake_store, keyword_store=fake_kw)
    results = r.retrieve("attention", top_k=5)

    ids = [c["id"] for c in results]
    assert "A_0" in ids
    assert "B_0" in ids
    fake_kw.query.assert_called_once()


def test_retriever_applies_reranker_when_present():
    fake_store = MagicMock()
    fake_store.query.return_value = [
        {"id": "A_0", "document": "x", "metadata": {"arxiv_id": "A"}, "distance": 0.1},
        {"id": "C_0", "document": "z", "metadata": {"arxiv_id": "C"}, "distance": 0.2},
    ]
    fake_kw = MagicMock()
    fake_kw.query.return_value = []
    fake_reranker = MagicMock()
    fake_reranker.rerank.side_effect = lambda q, c, top_k: list(reversed(c))[:top_k]

    r = Retriever(embedder=_embedder(), store=fake_store,
                  keyword_store=fake_kw, reranker=fake_reranker)
    results = r.retrieve("q", top_k=5)
    fake_reranker.rerank.assert_called_once()
    assert results[0]["id"] == "C_0"
