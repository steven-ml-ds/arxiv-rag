from app.rag.fusion import reciprocal_rank_fusion


def test_rrf_rewards_agreement_across_rankings():
    vec = ["a", "b", "c"]
    kw = ["b", "a", "d"]
    fused = reciprocal_rank_fusion([vec, kw])
    assert set(fused[:2]) == {"a", "b"}
    assert "c" in fused and "d" in fused


def test_rrf_handles_empty_rankings():
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_ranking_preserves_order():
    assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]
