from unittest.mock import MagicMock

from app.rag.generator import Generator, build_context_block


def test_build_context_block_formats_chunks():
    chunks = [
        {"document": "chunk A text", "metadata": {"arxiv_id": "2401.00001", "title": "First"}},
        {"document": "chunk B text", "metadata": {"arxiv_id": "2402.00002", "title": "Second"}},
    ]
    out = build_context_block(chunks)
    assert "arxiv_id: 2401.00001" in out
    assert "title: First" in out
    assert "chunk A text" in out
    assert "arxiv_id: 2402.00002" in out
    assert "chunk B text" in out


def test_generator_calls_claude_with_context():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="The answer based on context.")]
    )

    g = Generator(client=fake_client, model="claude-sonnet-4-6")
    chunks = [
        {"document": "FlashAttention is a fast attention algorithm.",
         "metadata": {"arxiv_id": "2205.14135", "title": "FlashAttention"}},
    ]
    answer = g.generate(question="What is FlashAttention?", chunks=chunks)

    assert answer == "The answer based on context."
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    user_msg = call_kwargs["messages"][0]["content"]
    assert "FlashAttention is a fast attention algorithm." in user_msg
    assert "What is FlashAttention?" in user_msg
