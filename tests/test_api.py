from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Build a TestClient with all heavy deps mocked at the dependency layer."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    from app import main
    from app.api import chat as chat_api

    # Override the FastAPI dependency that returns a Retriever
    fake_retriever = MagicMock()
    fake_retriever.retrieve.return_value = [
        {"id": "x", "document": "stub chunk", "metadata": {"arxiv_id": "2401.00001", "title": "Stub"}, "distance": 0.0}
    ]
    fake_generator = MagicMock()
    fake_generator.generate.return_value = "stubbed answer"

    main.app.dependency_overrides[chat_api.get_retriever] = lambda: fake_retriever
    main.app.dependency_overrides[chat_api.get_generator] = lambda: fake_generator

    fake_log = MagicMock()
    fake_rewriter = MagicMock()
    fake_rewriter.rewrite.side_effect = lambda q, history: q  # identity
    main.app.dependency_overrides[chat_api.get_query_log] = lambda: fake_log
    main.app.dependency_overrides[chat_api.get_query_rewriter] = lambda: fake_rewriter

    with TestClient(main.app) as c:
        yield c

    main.app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_returns_answer_and_sources(client):
    r = client.post("/chat", json={"q": "What is FlashAttention?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "stubbed answer"
    assert body["sources"][0]["arxiv_id"] == "2401.00001"
    assert body["sources"][0]["title"] == "Stub"


def test_chat_rejects_empty_query(client):
    r = client.post("/chat", json={"q": ""})
    assert r.status_code == 422


def test_root_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "arxiv-rag" in r.text.lower()
