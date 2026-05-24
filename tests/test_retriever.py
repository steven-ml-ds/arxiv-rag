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
