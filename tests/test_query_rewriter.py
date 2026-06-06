from unittest.mock import MagicMock

from app.rag.query_rewriter import QueryRewriter


def test_rewrite_returns_standalone_query():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="  FlashAttention v3 improvements  ")]
    )
    r = QueryRewriter(client=fake_client, model="claude-haiku-4-5")
    out = r.rewrite("how about v3?", [
        {"role": "user", "content": "what is FlashAttention?"},
        {"role": "assistant", "content": "It is a fast attention algorithm."},
    ])
    assert out == "FlashAttention v3 improvements"
    kwargs = fake_client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    prompt = kwargs["messages"][0]["content"]
    assert "how about v3?" in prompt
    assert "what is FlashAttention?" in prompt


def test_rewrite_empty_history_returns_question_unchanged():
    fake_client = MagicMock()
    r = QueryRewriter(client=fake_client, model="claude-haiku-4-5")
    assert r.rewrite("what is X?", []) == "what is X?"
    fake_client.messages.create.assert_not_called()
