from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    section: str


_SECTION_RE = re.compile(
    r"^[ \t]*(?:\d+\.?\s+)?("
    r"abstract|introduction|background|related work|"
    r"method(?:s|ology)?|approach|model|experiment(?:s)?|"
    r"result(?:s)?|evaluation|analysis|discussion|conclusion(?:s)?|"
    r"references|bibliography|acknowledg|appendix"
    r")\b",
    re.IGNORECASE | re.MULTILINE,
)
_DROP_PREFIXES = ("references", "bibliography", "acknowledg")


def _normalise_section(name: str) -> str:
    n = name.strip().lower()
    if n in ("methods", "methodology"):
        return "method"
    if n == "conclusions":
        return "conclusion"
    if n == "experiment":
        return "experiments"
    if n == "result":
        return "results"
    return n


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split paper text into (section_name, body) pairs by detected headers.

    Text before the first detected header is returned as 'preamble'. If no
    headers are found at all, the whole text is one ('body', text) pair.
    """
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [("body", text)]

    out: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        out.append(("preamble", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        name = _normalise_section(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((name, text[start:end]))
    return out


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Naive fixed-size chunker with overlap.

    Used by chunk_sections as the within-section splitter.
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


def chunk_sections(text: str, size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Section-aware chunker: drop References/bibliography/acknowledgments,
    then fixed-size-chunk each remaining section, tagging chunks with section."""
    out: list[Chunk] = []
    for name, body in split_sections(text):
        if any(name.startswith(p) for p in _DROP_PREFIXES):
            continue
        for piece in chunk_text(body, size=size, overlap=overlap):
            out.append(Chunk(text=piece, section=name))
    return out
