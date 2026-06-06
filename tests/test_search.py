from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    from app import main
    from app.api import search as search_api

    fake_retriever = MagicMock()
    fake_retriever.retrieve.return_value = [
        {
            "id": "arxiv_2401.00001_chunk_2",
            "document": "FlashAttention computes exact attention with less memory traffic.",
            "metadata": {
                "arxiv_id": "2401.00001",
                "title": "FlashAttention",
                "authors": "Tri Dao",
                "year": 2022,
                "chunk_index": 2,
            },
            "distance": 0.05,
        },
        {
            "id": "arxiv_2401.00001_chunk_4",
            "document": "The algorithm tiles attention to avoid materializing the full matrix.",
            "metadata": {
                "arxiv_id": "2401.00001",
                "title": "FlashAttention",
                "authors": "Tri Dao",
                "year": 2022,
                "chunk_index": 4,
            },
            "distance": 0.08,
        },
        {
            "id": "arxiv_2401.00002_chunk_0",
            "document": "Another paper discusses efficient transformer training.",
            "metadata": {
                "arxiv_id": "2401.00002",
                "title": "Other Attention Paper",
                "authors": "Ada Lovelace",
                "year": 2024,
                "chunk_index": 0,
            },
            "distance": 0.12,
        },
    ]

    main.app.dependency_overrides[search_api.get_retriever] = lambda: fake_retriever
    with TestClient(main.app) as c:
        c.fake_retriever = fake_retriever
        yield c

    main.app.dependency_overrides.clear()


def test_search_returns_grouped_article_results(client):
    r = client.post("/search", json={"q": "flashattention", "top_k": 10})

    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    first = body["results"][0]
    assert first["arxiv_id"] == "2401.00001"
    assert first["title"] == "FlashAttention"
    assert first["authors"] == "Tri Dao"
    assert first["year"] == 2022
    assert first["score"] == 0.95
    assert [c["chunk_id"] for c in first["matched_chunks"]] == [
        "arxiv_2401.00001_chunk_2",
        "arxiv_2401.00001_chunk_4",
    ]
    assert first["matched_chunks"][0]["chunk_index"] == 2
    assert first["matched_chunks"][0]["distance"] == 0.05
    assert "FlashAttention computes exact attention" in first["matched_chunks"][0]["snippet"]


def test_search_fetches_extra_chunks_before_grouping(client):
    r = client.post("/search", json={"q": "flashattention", "top_k": 2})

    assert r.status_code == 200
    from app.api.search import MAX_MATCHED_CHUNKS

    client.fake_retriever.retrieve.assert_called_once_with(
        "flashattention", top_k=6, max_per_paper=MAX_MATCHED_CHUNKS
    )


def test_search_orders_articles_by_best_distance(client):
    r = client.post("/search", json={"q": "flashattention", "top_k": 10})

    assert r.status_code == 200
    results = r.json()["results"]
    assert [item["arxiv_id"] for item in results] == ["2401.00001", "2401.00002"]


def test_search_limits_matched_chunks_to_three(client):
    from app.api.search import group_chunks_by_article

    chunks = [
        {
            "id": f"arxiv_2401.00001_chunk_{i}",
            "document": f"chunk {i}",
            "metadata": {
                "arxiv_id": "2401.00001",
                "title": "FlashAttention",
                "authors": "Tri Dao",
                "year": 2022,
                "chunk_index": i,
            },
            "distance": i / 100,
        }
        for i in range(5)
    ]

    results = group_chunks_by_article(chunks, top_k=1)

    assert len(results) == 1
    assert len(results[0].matched_chunks) == 3
    assert [c.chunk_id for c in results[0].matched_chunks] == [
        "arxiv_2401.00001_chunk_0",
        "arxiv_2401.00001_chunk_1",
        "arxiv_2401.00001_chunk_2",
    ]


def test_search_uses_safe_defaults_for_missing_metadata():
    from app.api.search import group_chunks_by_article

    results = group_chunks_by_article(
        [
            {
                "id": "chunk-1",
                "document": "A chunk without complete metadata.",
                "metadata": {},
                "distance": 0.2,
            }
        ],
        top_k=1,
    )

    assert results[0].arxiv_id == "unknown"
    assert results[0].title == "Unknown title"
    assert results[0].authors == ""
    assert results[0].year is None
    assert results[0].matched_chunks[0].chunk_index == 0


def test_search_rejects_empty_query(client):
    r = client.post("/search", json={"q": ""})

    assert r.status_code == 422


def test_search_allows_three_chunks_per_article_through_real_retriever(monkeypatch):
    """Regression for the per-paper dedup interaction.

    Drives the REAL Retriever (not a mocked .retrieve) so the diversity cap is
    actually exercised. The store returns 5 chunks from one paper; /search must
    still surface MAX_MATCHED_CHUNKS (3) of them. Under the bare retriever
    default (max_per_paper=2) this would collapse to 2.
    """
    import numpy as np

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from app import main
    from app.api import search as search_api
    from app.rag.retriever import Retriever

    fake_store = MagicMock()
    fake_store.query.return_value = [
        {
            "id": f"arxiv_2401.00001_chunk_{i}",
            "document": f"chunk {i}",
            "metadata": {
                "arxiv_id": "2401.00001",
                "title": "FlashAttention",
                "authors": "Tri Dao",
                "year": 2022,
                "chunk_index": i,
            },
            "distance": i / 100,
        }
        for i in range(5)
    ]
    fake_embedder = MagicMock()
    fake_embedder.embed_query.return_value = np.zeros(1024, dtype=np.float32)
    fake_kw = MagicMock()
    fake_kw.query.return_value = []
    real_retriever = Retriever(
        embedder=fake_embedder, store=fake_store, keyword_store=fake_kw
    )

    main.app.dependency_overrides[search_api.get_retriever] = lambda: real_retriever
    try:
        with TestClient(main.app) as c:
            r = c.post("/search", json={"q": "flashattention", "top_k": 5})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 1
        assert len(results[0]["matched_chunks"]) == 3
    finally:
        main.app.dependency_overrides.clear()
