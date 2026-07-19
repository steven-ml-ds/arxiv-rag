import json
import random
from unittest.mock import MagicMock

from eval.generate_questions import generate_multi_hop, sample_papers


def test_sample_papers_dedupes_by_arxiv_id():
    store = MagicMock()
    store.scan.side_effect = [
        [
            {"id": "A_0", "document": "chunk a0", "metadata": {"arxiv_id": "A", "title": "TA"}},
            {"id": "A_1", "document": "chunk a1", "metadata": {"arxiv_id": "A", "title": "TA"}},
            {"id": "B_0", "document": "chunk b0", "metadata": {"arxiv_id": "B", "title": "TB"}},
        ],
        [],
    ]

    papers = sample_papers(store)

    assert set(papers) == {"A", "B"}
    assert papers["A"]["excerpt"] == "chunk a0"  # first chunk wins


def test_generate_multi_hop_pairs_and_parses():
    papers = {
        "A": {"title": "TA", "excerpt": "ea"},
        "B": {"title": "TB", "excerpt": "eb"},
        "C": {"title": "TC", "excerpt": "ec"},
    }
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=json.dumps(
            {"question": "How do they differ?", "rationale": "needs both"}
        ))]
    )

    out = generate_multi_hop(client, "model", papers, n=2, rng=random.Random(0))

    assert len(out) == 2
    for item in out:
        assert item["category"] == "multi_hop"
        assert len(item["expected_arxiv_ids"]) == 2
        assert item["question"] == "How do they differ?"


def test_generate_multi_hop_skips_malformed_and_continues():
    papers = {"A": {"title": "TA", "excerpt": "ea"}, "B": {"title": "TB", "excerpt": "eb"},
              "C": {"title": "TC", "excerpt": "ec"}, "D": {"title": "TD", "excerpt": "ed"}}
    good = json.dumps({"question": "q?", "rationale": "r"})
    client = MagicMock()
    client.messages.create.side_effect = [
        MagicMock(content=[MagicMock(text="not json")]),
        MagicMock(content=[MagicMock(text=good)]),
    ]

    out = generate_multi_hop(client, "model", papers, n=1, rng=random.Random(0))

    assert len(out) == 1
