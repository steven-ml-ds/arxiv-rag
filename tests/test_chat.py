from app.api.chat import cited_sources


def _chunk(arxiv_id, chunk_id, title="T"):
    return {"id": chunk_id, "metadata": {"arxiv_id": arxiv_id, "title": title}}


def test_cited_sources_keeps_only_referenced_papers():
    chunks = [_chunk("2401.00001", "a_0"), _chunk("2402.00002", "b_0")]
    answer = "FlashAttention is fast [arxiv:2401.00001]."
    sources = cited_sources(answer, chunks)
    assert [s.arxiv_id for s in sources] == ["2401.00001"]


def test_cited_sources_falls_back_to_all_when_none_cited():
    chunks = [_chunk("2401.00001", "a_0"), _chunk("2402.00002", "b_0")]
    sources = cited_sources("No citation markers here.", chunks)
    assert {s.arxiv_id for s in sources} == {"2401.00001", "2402.00002"}


def test_cited_sources_dedupes_by_arxiv_id():
    chunks = [_chunk("2401.00001", "a_0"), _chunk("2401.00001", "a_1")]
    sources = cited_sources("see [arxiv:2401.00001]", chunks)
    assert len(sources) == 1
