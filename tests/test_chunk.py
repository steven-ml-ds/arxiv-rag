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
