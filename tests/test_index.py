from pathlib import Path

from app.stores.vector_store import VectorStore
from pipeline.chunk import Chunk
from pipeline.fetch_arxiv import Paper
from pipeline.index import index_paper


def _paper() -> Paper:
    return Paper(
        arxiv_id="2401.00001",
        title="Test",
        authors=["A"],
        published_year=2024,
        pdf_path=Path("/tmp/x.pdf"),
    )


def test_index_paper_writes_all_chunks(tmp_path, fake_vec_1024):
    store = VectorStore(path=str(tmp_path / "chroma"), collection="test_idx")
    chunks = [
        Chunk(text="chunk a", section="abstract"),
        Chunk(text="chunk b", section="method"),
        Chunk(text="chunk c", section="results"),
    ]
    embeddings = [fake_vec_1024(i).tolist() for i in range(3)]

    index_paper(_paper(), chunks, embeddings, store)

    assert store.count() == 3
    res = store.query(query_embedding=fake_vec_1024(0).tolist(), top_k=1)
    assert res[0]["id"] == "arxiv_2401.00001_chunk_0"
    assert res[0]["metadata"]["arxiv_id"] == "2401.00001"
    assert res[0]["metadata"]["title"] == "Test"
    assert res[0]["metadata"]["year"] == 2024
    assert res[0]["metadata"]["chunk_index"] == 0
    assert res[0]["metadata"]["section"] == "abstract"


def test_index_paper_rejects_mismatched_lengths(tmp_path, fake_vec_1024):
    import pytest

    store = VectorStore(path=str(tmp_path / "chroma"), collection="test_idx_mm")
    chunks = [Chunk(text="a", section="abstract"), Chunk(text="b", section="method")]
    with pytest.raises(ValueError):
        index_paper(_paper(), chunks, [fake_vec_1024(0).tolist()], store)
