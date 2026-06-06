from pathlib import Path
from unittest.mock import patch

from pipeline.parse_pdf import pdf_to_text


def test_pdf_to_text_returns_string():
    """Blank PDF returns an empty (or whitespace) string without crashing."""
    result = pdf_to_text(Path("tests/fixtures/tiny.pdf"))
    assert isinstance(result, str)


def test_pdf_to_text_joins_pages():
    """Multi-page extraction joins page text with double newline separators."""
    with patch("pipeline.parse_pdf.PdfReader") as mock_reader:
        mock_reader.return_value.pages = [
            type("P", (), {"extract_text": lambda self: "page one"})(),
            type("P", (), {"extract_text": lambda self: "page two"})(),
        ]
        out = pdf_to_text(Path("ignored.pdf"))
        assert out == "page one\n\npage two"


def test_pdf_to_text_handles_none_returning_pages():
    """Some pages return None from extract_text — must not crash."""
    with patch("pipeline.parse_pdf.PdfReader") as mock_reader:
        mock_reader.return_value.pages = [
            type("P", (), {"extract_text": lambda self: None})(),
            type("P", (), {"extract_text": lambda self: "real text"})(),
        ]
        out = pdf_to_text(Path("ignored.pdf"))
        assert out == "real text"


def _has_surrogate(s: str) -> bool:
    return any(0xD800 <= ord(c) <= 0xDFFF for c in s)


def test_sanitize_text_strips_lone_surrogates():
    """pypdf can emit lone UTF-16 surrogates from broken font encodings; the
    Rust tokenizer rejects them (TextEncodeInput TypeError). Strip them."""
    from pipeline.parse_pdf import sanitize_text

    bad = "good text \ud83d more \udfff text"
    out = sanitize_text(bad)
    assert not _has_surrogate(out)
    # round-trip must be valid UTF-8 (what the tokenizer requires)
    out.encode("utf-8")
    assert "good text" in out and "more" in out and "text" in out


def test_pdf_to_text_sanitizes_surrogates_from_pages():
    with patch("pipeline.parse_pdf.PdfReader") as mock_reader:
        mock_reader.return_value.pages = [
            type("P", (), {"extract_text": lambda self: "clean \ud83d page"})(),
        ]
        out = pdf_to_text(Path("ignored.pdf"))
        assert not _has_surrogate(out)
        out.encode("utf-8")
