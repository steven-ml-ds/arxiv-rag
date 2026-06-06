from eval.metrics import citation_accuracy, hit_at_k


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
