from __future__ import annotations

import json
from typing import Any, Callable

from app.rag.generator import build_context_block

JUDGE_MODEL = "claude-haiku-4-5"

JUDGE_SYSTEM = (
    "You are a strict fact-checking judge for a RAG system. You receive the "
    "context that was shown to an answering model, and the answer it produced. "
    "Break the answer into its individual factual claims, then check each claim "
    "against the context ONLY (ignore your own knowledge). Reply with strict JSON "
    '{"claims": [{"text": "<claim>", "verdict": "supported"|"unsupported"}]} '
    "and nothing else. A refusal or an answer with no factual claims gets "
    '{"claims": []}.'
)


def _parse_claims(raw: str) -> list[dict[str, str]]:
    """Parse the judge's JSON, tolerating a fenced code block wrapper."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.index("{"):]
    data = json.loads(text)
    claims = data["claims"]
    if not isinstance(claims, list):
        raise ValueError("claims is not a list")
    for c in claims:
        if c["verdict"] not in ("supported", "unsupported"):
            raise ValueError(f"bad verdict: {c['verdict']!r}")
    return claims


def judge_faithfulness(
    question: str,
    chunks: list[dict[str, Any]],
    answer: str,
    client,
    model: str = JUDGE_MODEL,
) -> dict[str, Any]:
    """Score one answer's faithfulness to its retrieval context.

    Returns {"faithfulness": supported/total, "judge_claims": total} — or
    faithfulness 1.0 for claim-free answers (nothing to hallucinate), or
    {"judge_error": ...} if the judge output stays unparseable after one retry.
    """
    user = (
        f"Context:\n{build_context_block(chunks)}\n\n"
        f"Question: {question}\n\n"
        f"Answer to check:\n{answer}"
    )
    last_err: Exception | None = None
    for _ in range(2):  # one retry on malformed JSON
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        try:
            claims = _parse_claims(resp.content[0].text)
        except (ValueError, KeyError, json.JSONDecodeError) as err:
            last_err = err
            continue
        if not claims:
            return {"faithfulness": 1.0, "judge_claims": 0}
        supported = sum(1 for c in claims if c["verdict"] == "supported")
        return {"faithfulness": supported / len(claims), "judge_claims": len(claims)}
    return {"judge_error": str(last_err)}


def make_judge(client, model: str = JUDGE_MODEL) -> Callable[..., dict[str, Any]]:
    """Bind a client into the (question, chunks, answer) callable evaluate() expects."""

    def judge(question: str, chunks: list[dict[str, Any]], answer: str) -> dict[str, Any]:
        return judge_faithfulness(question, chunks, answer, client, model=model)

    return judge
