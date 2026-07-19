from __future__ import annotations

import re

_CITE_RE = re.compile(r"\[arxiv:([^\]]+)\]")

# Must stay in sync with the sentinel in app.rag.generator.SYSTEM_PROMPT.
REFUSAL_SENTINEL = "I don't have enough information in the indexed papers to answer that."


def hit_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> bool:
    """True if any expected arxiv_id appears in the top-k retrieved ids."""
    topk = set(retrieved_ids[:k])
    return any(e in topk for e in expected_ids)


def citation_accuracy(answer: str, expected_ids: list[str]) -> float:
    """Precision of inline [arxiv:<id>] citations against the expected set.

    Returns |cited & expected| / |cited|, or 0.0 when the answer cites nothing.
    """
    cited = set(_CITE_RE.findall(answer))
    if not cited:
        return 0.0
    correct = len(cited & set(expected_ids))
    return correct / len(cited)


def all_hit_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> bool:
    """True only if EVERY expected arxiv_id appears in the top-k retrieved ids.

    Multi-hop questions need evidence from all expected papers, so any-hit
    (hit_at_k) would over-credit retrieval that finds just one of them.
    """
    if not expected_ids:
        return False
    topk = set(retrieved_ids[:k])
    return all(e in topk for e in expected_ids)


def is_refusal(answer: str) -> bool:
    """True if the answer contains the generator's refusal sentinel.

    Substring match: the model sometimes wraps the sentinel in extra prose
    even though the prompt asks for the exact sentence.
    """
    return REFUSAL_SENTINEL in answer


def citation_grounding(answer: str, retrieved_ids: list[str]) -> float:
    """Share of cited ids that were actually retrieved.

    Citing a paper that never appeared in the context is fabrication, whatever
    the citation's topical accuracy. Returns 1.0 when the answer cites nothing
    (no citations means nothing fabricated).
    """
    cited = set(_CITE_RE.findall(answer))
    if not cited:
        return 1.0
    return len(cited & set(retrieved_ids)) / len(cited)
