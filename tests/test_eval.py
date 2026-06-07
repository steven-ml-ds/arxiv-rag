from unittest.mock import MagicMock

from eval.metrics import citation_accuracy, hit_at_k
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
