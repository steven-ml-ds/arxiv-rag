from app.stores.keyword_store import KeywordStore


def test_add_and_query_returns_matching_ids(tmp_path):
    ks = KeywordStore(str(tmp_path / "kw.db"))
    ks.add(
        ids=["a_0", "b_0"],
        documents=["flash attention is fast", "mamba is a state space model"],
    )
    hits = ks.query("attention", top_k=5)
    ids = [h["id"] for h in hits]
    assert "a_0" in ids
    assert "b_0" not in ids


def test_query_empty_or_punctuation_returns_empty(tmp_path):
    ks = KeywordStore(str(tmp_path / "kw.db"))
    ks.add(ids=["a_0"], documents=["hello world"])
    assert ks.query("", top_k=5) == []
    assert ks.query("?!--", top_k=5) == []


def test_add_is_idempotent_on_reindex(tmp_path):
    db = str(tmp_path / "kw.db")
    ks = KeywordStore(db)
    ks.add(ids=["a_0"], documents=["attention mechanism"])
    ks.add(ids=["a_0"], documents=["attention mechanism revised"])
    hits = ks.query("attention", top_k=5)
    assert len([h for h in hits if h["id"] == "a_0"]) == 1


def test_keyword_store_usable_across_threads(tmp_path):
    """Regression: the singleton store is created in one thread but queried from
    FastAPI's threadpool workers — sqlite must allow cross-thread use."""
    import threading

    ks = KeywordStore(str(tmp_path / "kw.db"))
    ks.add(ids=["a_0"], documents=["attention is all you need"])

    out: dict = {}

    def worker():
        try:
            out["hits"] = ks.query("attention", top_k=5)
        except Exception as e:  # noqa: BLE001
            out["error"] = repr(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert "error" not in out, out.get("error")
    assert [h["id"] for h in out["hits"]] == ["a_0"]
