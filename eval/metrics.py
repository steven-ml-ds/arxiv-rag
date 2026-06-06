from __future__ import annotations

import re

_CITE_RE = re.compile(r"\[arxiv:([^\]]+)\]")


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
