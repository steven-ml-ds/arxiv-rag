from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def sanitize_text(text: str) -> str:
    """Drop characters that aren't valid Unicode scalars.

    pypdf can emit lone UTF-16 surrogate code points (U+D800–U+DFFF) from
    broken font/CMap encodings. They are legal Python ``str`` characters but
    invalid Unicode, and the Rust ``tokenizers`` backend rejects any batch
    containing them with ``TextEncodeInput must be Union[...]``. Round-tripping
    through UTF-8 with errors="ignore" strips exactly those characters.
    """
    return text.encode("utf-8", "ignore").decode("utf-8")


def pdf_to_text(pdf_path: Path) -> str:
    """Extract text from every page of a PDF and join with double newlines.

    M1 uses pypdf for simplicity. M2 swaps to `unstructured` for table-aware
    parsing and section labels. Output is sanitized so downstream embedding,
    storage, and reranking never see invalid Unicode.
    """
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return sanitize_text("\n\n".join(parts))
