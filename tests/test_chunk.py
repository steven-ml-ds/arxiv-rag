from pipeline.chunk import chunk_text


def test_short_text_returns_single_chunk():
    text = "short body"
    assert chunk_text(text, size=100, overlap=10) == ["short body"]


def test_chunks_have_overlap():
    text = "a" * 250
    chunks = chunk_text(text, size=100, overlap=20)
    # With size=100 overlap=20: starts at 0, 80, 160 → 3 chunks
    assert len(chunks) == 3
    assert all(len(c) <= 100 for c in chunks)
    # Last 20 chars of chunk[0] match first 20 chars of chunk[1]
    assert chunks[0][-20:] == chunks[1][:20]


def test_overlap_must_be_smaller_than_size():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("x" * 100, size=50, overlap=50)


def test_empty_input_returns_empty():
    assert chunk_text("", size=100, overlap=10) == []


from pipeline.chunk import Chunk, chunk_sections, split_sections


def test_split_sections_detects_headers():
    text = (
        "Title and authors\n"
        "Abstract\nWe present a method.\n"
        "1 Introduction\nPrior work exists.\n"
        "References\n[1] Someone et al.\n"
    )
    sections = dict(split_sections(text))
    assert "abstract" in sections
    assert "introduction" in sections
    assert "references" in sections
    assert "We present a method." in sections["abstract"]


def test_split_sections_no_headers_is_single_body():
    assert split_sections("just flat text") == [("body", "just flat text")]


def test_chunk_sections_drops_references_and_tags_section():
    text = (
        "Abstract\n" + "a" * 50 + "\n"
        "Introduction\n" + "b" * 50 + "\n"
        "References\n" + "c" * 500 + "\n"
    )
    chunks = chunk_sections(text, size=100, overlap=10)
    sections = {c.section for c in chunks}
    assert "references" not in sections
    assert "abstract" in sections
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(len(c.text) <= 100 for c in chunks)
