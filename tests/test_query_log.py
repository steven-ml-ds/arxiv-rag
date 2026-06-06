from app.stores.query_log import QueryLog


def test_log_then_recent_roundtrip(tmp_path):
    ql = QueryLog(str(tmp_path / "q.db"))
    ql.log("orig q", "rewritten q", ["a_0", "b_1"], 1234.5, 100, 42)
    rows = ql.recent(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["question"] == "orig q"
    assert row["rewritten"] == "rewritten q"
    assert row["doc_ids"] == "a_0,b_1"
    assert row["latency_ms"] == 1234.5
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 42


def test_recent_orders_newest_first(tmp_path):
    ql = QueryLog(str(tmp_path / "q.db"))
    ql.log("first", "first", [], 1.0, None, None)
    ql.log("second", "second", [], 2.0, None, None)
    rows = ql.recent(limit=10)
    assert rows[0]["question"] == "second"


def test_query_log_usable_across_threads(tmp_path):
    import threading

    ql = QueryLog(str(tmp_path / "q.db"))
    out: dict = {}

    def worker():
        try:
            ql.log("q", "q", ["x_0"], 5.0, 1, 1)
            out["rows"] = ql.recent(limit=1)
        except Exception as e:  # noqa: BLE001
            out["error"] = repr(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert "error" not in out, out.get("error")
    assert out["rows"][0]["question"] == "q"
