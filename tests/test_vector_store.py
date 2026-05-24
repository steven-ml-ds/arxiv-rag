from app.stores.vector_store import VectorStore


def test_add_and_query_roundtrip(tmp_path, fake_vec_1024):
    store = VectorStore(path=str(tmp_path / "chroma"), collection="test_papers")

    v1 = fake_vec_1024(1)
    v2 = fake_vec_1024(2)
    store.add(
        ids=["id1", "id2"],
        embeddings=[v1.tolist(), v2.tolist()],
        documents=["chunk one", "chunk two"],
        metadatas=[{"arxiv_id": "2401.00001"}, {"arxiv_id": "2401.00002"}],
    )

    assert store.count() == 2

    # Querying with v1 should return id1 first (cosine == 1.0 with itself)
    results = store.query(query_embedding=v1.tolist(), top_k=2)
    assert results[0]["id"] == "id1"
    assert results[0]["document"] == "chunk one"
    assert results[0]["metadata"]["arxiv_id"] == "2401.00001"
    assert len(results) == 2


def test_persistence_across_instances(tmp_path, fake_vec_1024):
    path = str(tmp_path / "chroma")
    s1 = VectorStore(path=path, collection="test_papers")
    s1.add(
        ids=["id1"],
        embeddings=[fake_vec_1024(1).tolist()],
        documents=["chunk"],
        metadatas=[{"arxiv_id": "2401.00001"}],
    )

    s2 = VectorStore(path=path, collection="test_papers")
    assert s2.count() == 1
