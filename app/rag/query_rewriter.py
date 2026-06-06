from __future__ import annotations

from typing import Any

from anthropic import Anthropic

_INSTRUCTION = (
    "Rewrite the user's follow-up question as a standalone search query for a "
    "vector database of AI/ML papers. Resolve pronouns and references using the "
    "conversation. Output ONLY the query text, nothing else."
)


class QueryRewriter:
    """Uses Claude Haiku to turn a conversational follow-up into a standalone query."""

    def __init__(self, client: Anthropic, model: str, max_tokens: int = 128):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def rewrite(self, question: str, history: list[dict[str, Any]]) -> str:
        if not history:
            return question  # first turn needs no rewriting; skip the API call
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        prompt = (
            f"{_INSTRUCTION}\n\nConversation:\n{convo}\n\nFollow-up: {question}\n\nQuery:"
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
