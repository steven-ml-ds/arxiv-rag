from unittest.mock import MagicMock

from app.rag.reranker import Reranker


def test_rerank_orders_by_model_score_and_truncates():
    r = Reranker.__new__(Reranker)            # bypass model download
    r.model = MagicMock()
    r.model.predict.return_value = [0.1, 0.9, 0.5]
    candidates = [
        {"id": "a", "document": "da"},
        {"id": "b", "document": "db"},
        {"id": "c", "document": "dc"},
    ]
    out = r.rerank("q", candidates, top_k=2)
    assert [c["id"] for c in out] == ["b", "c"]


def test_rerank_empty_returns_empty():
    r = Reranker.__new__(Reranker)
    r.model = MagicMock()
    assert r.rerank("q", [], top_k=5) == []
