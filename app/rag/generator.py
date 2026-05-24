from __future__ import annotations

from typing import Any

from anthropic import Anthropic

SYSTEM_PROMPT = (
    "You are a research assistant who answers questions about AI/ML papers. "
    "Use ONLY the provided context. If the context is insufficient, say so. "
    "Be precise and concise. Cite papers inline using [arxiv:<id>] when you "
    "reference them."
)


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


class Generator:
    """Builds the RAG prompt and calls Claude for a (non-streaming) answer."""

    def __init__(self, client: Anthropic, model: str, max_tokens: int = 1024):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, question: str, chunks: list[dict[str, Any]]) -> str:
        context = build_context_block(chunks)
        user_content = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using the context above."
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return resp.content[0].text
