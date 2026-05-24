from __future__ import annotations


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Naive fixed-size chunker with overlap.

    M1 baseline; M2 replaces this with section-aware semantic chunking.
    """
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be < size ({size})")
    if not text:
        return []

    chunks: list[str] = []
    step = size - overlap
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks
