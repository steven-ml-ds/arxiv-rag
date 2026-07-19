import json
from unittest.mock import MagicMock

from eval.judge import judge_faithfulness, make_judge


def _client_returning(*texts: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.side_effect = [
        MagicMock(content=[MagicMock(text=t)]) for t in texts
    ]
    return client


CHUNKS = [{"document": "ctx", "metadata": {"arxiv_id": "A", "title": "T"}}]


def test_judge_scores_supported_ratio():
    payload = json.dumps({"claims": [
        {"text": "c1", "verdict": "supported"},
        {"text": "c2", "verdict": "supported"},
        {"text": "c3", "verdict": "unsupported"},
    ]})
    client = _client_returning(payload)

    out = judge_faithfulness("q", CHUNKS, "answer", client)

    assert out == {"faithfulness": 2 / 3, "judge_claims": 3}


def test_judge_full_score_for_claim_free_answer():
    client = _client_returning(json.dumps({"claims": []}))
    out = judge_faithfulness("q", CHUNKS, "I don't know.", client)
    assert out == {"faithfulness": 1.0, "judge_claims": 0}


def test_judge_tolerates_fenced_json():
    payload = '```json\n{"claims": [{"text": "c", "verdict": "supported"}]}\n```'
    client = _client_returning(payload)
    out = judge_faithfulness("q", CHUNKS, "answer", client)
    assert out["faithfulness"] == 1.0


def test_judge_retries_once_then_succeeds():
    good = json.dumps({"claims": [{"text": "c", "verdict": "supported"}]})
    client = _client_returning("not json at all", good)

    out = judge_faithfulness("q", CHUNKS, "answer", client)

    assert out["faithfulness"] == 1.0
    assert client.messages.create.call_count == 2


def test_judge_records_error_after_retry_exhausted():
    client = _client_returning("garbage", "still garbage")
    out = judge_faithfulness("q", CHUNKS, "answer", client)
    assert "judge_error" in out
    assert "faithfulness" not in out


def test_judge_rejects_bad_verdict():
    payload = json.dumps({"claims": [{"text": "c", "verdict": "maybe"}]})
    client = _client_returning(payload, payload)
    out = judge_faithfulness("q", CHUNKS, "answer", client)
    assert "judge_error" in out


def test_make_judge_binds_client():
    client = _client_returning(json.dumps({"claims": []}))
    judge = make_judge(client)
    out = judge("q", CHUNKS, "answer")
    assert out["faithfulness"] == 1.0
    assert client.messages.create.call_args.kwargs["model"] == "claude-haiku-4-5"
