from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def stream_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from app import main
    from app.api import chat as chat_api

    fake_retriever = MagicMock()
    fake_retriever.retrieve.return_value = [
        {"id": "arxiv_1_chunk_0", "document": "d", "metadata": {"arxiv_id": "1", "title": "T"}, "distance": 0.0}
    ]
    fake_generator = MagicMock()
    fake_generator.generate_stream.return_value = iter([
        {"type": "text", "text": "Hello "},
        {"type": "text", "text": "world"},
        {"type": "usage", "input_tokens": 5, "output_tokens": 2},
    ])
    fake_log = MagicMock()
    fake_rewriter = MagicMock()
    fake_rewriter.rewrite.return_value = "rewritten"

    main.app.dependency_overrides[chat_api.get_retriever] = lambda: fake_retriever
    main.app.dependency_overrides[chat_api.get_generator] = lambda: fake_generator
    main.app.dependency_overrides[chat_api.get_query_log] = lambda: fake_log
    main.app.dependency_overrides[chat_api.get_query_rewriter] = lambda: fake_rewriter

    with TestClient(main.app) as c:
        yield c, fake_log

    main.app.dependency_overrides.clear()


def test_chat_stream_emits_sse_tokens_and_logs(stream_client):
    client, fake_log = stream_client
    r = client.post("/chat/stream", json={"q": "hi"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "event: sources" in body
    assert "Hello " in body and "world" in body
    assert "event: done" in body
    fake_log.log.assert_called_once()
