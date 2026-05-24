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
