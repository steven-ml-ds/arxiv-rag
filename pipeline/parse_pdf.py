from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def pdf_to_text(pdf_path: Path) -> str:
    """Extract text from every page of a PDF and join with double newlines.

    M1 uses pypdf for simplicity. M2 swaps to `unstructured` for table-aware
    parsing and section labels.
    """
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)
