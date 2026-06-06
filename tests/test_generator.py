from unittest.mock import MagicMock

from app.rag.generator import Generator, build_context_block, build_messages


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


def test_generator_marks_system_prompt_for_caching():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="ok")]
    )
    g = Generator(client=fake_client, model="claude-sonnet-4-6")
    g.generate(question="q", chunks=[
        {"document": "d", "metadata": {"arxiv_id": "1", "title": "T"}}
    ])
    system = fake_client.messages.create.call_args.kwargs["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_build_messages_appends_context_user_turn():
    msgs = build_messages("What is X?", [
        {"document": "doc text", "metadata": {"arxiv_id": "1", "title": "T"}}
    ], history=None)
    assert msgs[-1]["role"] == "user"
    assert "doc text" in msgs[-1]["content"]
    assert "What is X?" in msgs[-1]["content"]


def test_build_messages_includes_recent_history_only():
    history = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    msgs = build_messages("now", [], history=history, max_turns=4)
    assert len(msgs) == 5
    assert msgs[0]["content"] == "q6"
    assert msgs[-1]["role"] == "user"


def test_generate_passes_history_to_client():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = MagicMock(content=[MagicMock(text="ok")])
    g = Generator(client=fake_client, model="claude-sonnet-4-6")
    g.generate("follow up", [{"document": "d", "metadata": {"arxiv_id": "1", "title": "T"}}],
               history=[{"role": "user", "content": "earlier"},
                        {"role": "assistant", "content": "reply"}])
    sent = fake_client.messages.create.call_args.kwargs["messages"]
    assert sent[0] == {"role": "user", "content": "earlier"}
    assert sent[1] == {"role": "assistant", "content": "reply"}
    assert sent[-1]["role"] == "user"


class _FakeStream:
    def __init__(self, texts, in_tok, out_tok):
        self._texts = texts
        self._in, self._out = in_tok, out_tok

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        return iter(self._texts)

    def get_final_message(self):
        return MagicMock(usage=MagicMock(input_tokens=self._in, output_tokens=self._out))


def test_generate_stream_yields_text_then_usage():
    fake_client = MagicMock()
    fake_client.messages.stream.return_value = _FakeStream(["Hello ", "world"], 12, 7)
    g = Generator(client=fake_client, model="claude-sonnet-4-6")

    events = list(g.generate_stream("q", [{"document": "d", "metadata": {"arxiv_id": "1", "title": "T"}}]))

    texts = [e["text"] for e in events if e["type"] == "text"]
    assert "".join(texts) == "Hello world"
    usage = [e for e in events if e["type"] == "usage"][-1]
    assert usage["input_tokens"] == 12
    assert usage["output_tokens"] == 7
