import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def dash_client(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from app import main
    from app.api import dashboard as dash_api

    fake_log = MagicMock()
    fake_log.recent.return_value = [
        {"ts": 1.0, "question": "q1", "rewritten": "q1", "doc_ids": "a_0",
         "latency_ms": 12.0, "input_tokens": 5, "output_tokens": 2}
    ]
    results_path = tmp_path / "eval_results.json"
    results_path.write_text(json.dumps({"hit_at_5": 0.8, "citation_accuracy": 0.9,
                                         "avg_latency_ms": 3200, "n": 20, "rows": []}))

    monkeypatch.setattr(dash_api, "_results_path", lambda: str(results_path))
    main.app.dependency_overrides[dash_api.get_query_log] = lambda: fake_log

    with TestClient(main.app) as c:
        yield c

    main.app.dependency_overrides.clear()


def test_stats_returns_queries_and_eval(dash_client):
    r = dash_client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["queries"][0]["question"] == "q1"
    assert body["eval"]["hit_at_5"] == 0.8


def test_dashboard_serves_html(dash_client):
    r = dash_client.get("/dashboard")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "dashboard" in r.text.lower()
