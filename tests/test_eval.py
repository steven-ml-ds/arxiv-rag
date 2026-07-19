from unittest.mock import MagicMock

from eval.metrics import (
    REFUSAL_SENTINEL,
    all_hit_at_k,
    citation_accuracy,
    citation_grounding,
    hit_at_k,
    is_refusal,
)
from eval.run_eval import evaluate


def test_hit_at_k_true_when_expected_in_topk():
    assert hit_at_k(["2205.14135", "x", "y", "z", "w"], ["2205.14135"], 5) is True


def test_hit_at_k_false_when_expected_below_k():
    assert hit_at_k(["a", "b", "c", "d", "e", "2205.14135"], ["2205.14135"], 5) is False


def test_hit_at_k_false_when_no_overlap():
    assert hit_at_k(["a", "b"], ["2205.14135"], 5) is False


def test_citation_accuracy_precision_of_cited():
    answer = "FlashAttention is fast [arxiv:2205.14135] unlike [arxiv:9999.99999]."
    assert citation_accuracy(answer, ["2205.14135"]) == 0.5


def test_citation_accuracy_zero_when_no_citations():
    assert citation_accuracy("no markers here", ["2205.14135"]) == 0.0


def test_all_hit_at_k_requires_every_expected_id():
    assert all_hit_at_k(["A", "B", "c"], ["A", "B"], 5) is True
    assert all_hit_at_k(["A", "x", "y", "z", "w", "B"], ["A", "B"], 5) is False
    assert all_hit_at_k(["A"], ["A", "B"], 5) is False


def test_all_hit_at_k_false_for_empty_expected():
    assert all_hit_at_k(["A"], [], 5) is False


def test_refusal_sentinel_matches_generator_prompt():
    from app.rag.generator import SYSTEM_PROMPT

    assert REFUSAL_SENTINEL in SYSTEM_PROMPT


def test_is_refusal_substring_tolerant():
    assert is_refusal(REFUSAL_SENTINEL) is True
    assert is_refusal(f"I'm sorry — {REFUSAL_SENTINEL} Try another question.") is True
    assert is_refusal("FlashAttention is fast [arxiv:2205.14135].") is False


def test_citation_grounding_penalizes_unretrieved_citations():
    answer = "Fast [arxiv:A] and slow [arxiv:B]."
    assert citation_grounding(answer, ["A", "x"]) == 0.5
    assert citation_grounding(answer, ["A", "B"]) == 1.0


def test_citation_grounding_full_when_no_citations():
    assert citation_grounding("no citations", ["A"]) == 1.0


def test_evaluate_aggregates_metrics():
    golden = [
        {"question": "q1", "expected_arxiv_ids": ["A"]},
        {"question": "q2", "expected_arxiv_ids": ["Z"]},  # will miss
    ]
    retriever = MagicMock()
    retriever.retrieve.side_effect = [
        [{"id": "A_0", "document": "d", "metadata": {"arxiv_id": "A", "title": "T"}}],
        [{"id": "B_0", "document": "d", "metadata": {"arxiv_id": "B", "title": "T"}}],
    ]
    generator = MagicMock()
    generator.generate.side_effect = ["answer [arxiv:A]", "answer [arxiv:B]"]

    res = evaluate(golden, retriever, generator, top_k=5)

    assert res["n"] == 2
    assert res["hit_at_5"] == 0.5
    assert res["citation_accuracy"] == 0.5
    assert res["avg_latency_ms"] >= 0.0
    assert len(res["rows"]) == 2


def _chunk(arxiv_id: str) -> dict:
    return {"id": f"{arxiv_id}_0", "document": "d",
            "metadata": {"arxiv_id": arxiv_id, "title": "T"}}


def test_evaluate_scores_by_category():
    golden = [
        {"question": "sh", "category": "single_hop", "expected_arxiv_ids": ["A"]},
        {"question": "mh-hit", "category": "multi_hop", "expected_arxiv_ids": ["A", "B"]},
        {"question": "mh-miss", "category": "multi_hop", "expected_arxiv_ids": ["A", "Z"]},
        {"question": "unans-good", "category": "unanswerable", "expected_arxiv_ids": []},
        {"question": "unans-bad", "category": "unanswerable", "expected_arxiv_ids": []},
    ]
    retriever = MagicMock()
    retriever.retrieve.side_effect = [
        [_chunk("A")],
        [_chunk("A"), _chunk("B")],
        [_chunk("A"), _chunk("B")],  # Z missing -> multi-hop all-hit fails
        [_chunk("X")],
        [_chunk("X")],
    ]
    generator = MagicMock()
    generator.generate.side_effect = [
        "answer [arxiv:A]",
        "answer [arxiv:A] and [arxiv:B]",
        "answer [arxiv:A]",
        REFUSAL_SENTINEL,
        "hallucinated answer [arxiv:X]",  # should have refused
    ]

    res = evaluate(golden, retriever, generator, top_k=5)

    cats = res["by_category"]
    assert cats["single_hop"]["n"] == 1
    assert cats["single_hop"]["hit_at_5"] == 1.0
    assert cats["multi_hop"]["hit_at_5"] == 0.5  # all-hit: one of two
    assert cats["unanswerable"]["refusal_accuracy"] == 0.5
    # Top-level metrics cover answerable rows only.
    assert res["n"] == 5
    assert res["hit_at_5"] == (1 + 1 + 0) / 3
    assert res["false_refusal_rate"] == 0.0


def test_evaluate_merges_judge_output():
    golden = [{"question": "q", "category": "single_hop", "expected_arxiv_ids": ["A"]}]
    retriever = MagicMock()
    retriever.retrieve.return_value = [_chunk("A")]
    generator = MagicMock()
    generator.generate.return_value = "answer [arxiv:A]"

    def fake_judge(question, chunks, answer):
        return {"faithfulness": 0.75, "judge_claims": 4}

    res = evaluate(golden, retriever, generator, top_k=5, judge=fake_judge)

    assert res["rows"][0]["faithfulness"] == 0.75
    assert res["by_category"]["single_hop"]["faithfulness"] == 0.75
