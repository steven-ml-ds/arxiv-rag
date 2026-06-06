from __future__ import annotations

from typing import Any

from anthropic import Anthropic

SYSTEM_PROMPT = (
    "You are a research assistant answering questions about AI/ML papers. "
    "Use ONLY the provided context - never rely on prior knowledge. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't have enough information in the indexed papers to answer that.\" "
    "After every claim, cite the supporting paper inline as [arxiv:<id>] using "
    "the arxiv_id shown in the context. Be precise and concise."
)

# NOTE: cache_control is wired correctly, but Claude only caches a block once it
# exceeds the minimum cacheable length (~1024 tokens for Sonnet/Opus). SYSTEM_PROMPT
# is far shorter, so caching is a no-op today. It starts paying off if/when we
# prepend a large stable prefix; the per-query retrieved context is deliberately
# NOT cached since it changes every call.
_SYSTEM_BLOCKS = [
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


def build_context_block(chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks into a single context string for the prompt."""
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        parts.append(
            f"[{i}] arxiv_id: {meta.get('arxiv_id', '?')} | "
            f"title: {meta.get('title', '?')}\n{c['document']}"
        )
    return "\n\n".join(parts)


def build_messages(
    question: str,
    chunks: list[dict[str, Any]],
    history: list[dict[str, Any]] | None = None,
    max_turns: int = 6,
) -> list[dict[str, Any]]:
    """Assemble the messages array: recent history + a context-grounded user turn."""
    context = build_context_block(chunks)
    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using the context above."
    )
    msgs: list[dict[str, Any]] = []
    if history:
        msgs.extend(history[-max_turns:])
    msgs.append({"role": "user", "content": user_content})
    return msgs


class Generator:
    """Builds the RAG prompt and calls Claude (non-streaming or streaming)."""

    def __init__(self, client: Anthropic, model: str, max_tokens: int = 1024):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_BLOCKS,
            messages=build_messages(question, chunks, history),
        )
        return resp.content[0].text

    def generate_stream(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
    ):
        """Yield {'type':'text','text':...} deltas, then one {'type':'usage',...}."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_BLOCKS,
            messages=build_messages(question, chunks, history),
        ) as stream:
            for text in stream.text_stream:
                yield {"type": "text", "text": text}
            usage = stream.get_final_message().usage
            yield {
                "type": "usage",
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }
