import numpy as np
from pipeline.embed import Embedder


def test_embedder_returns_correct_shape():
    e = Embedder("BAAI/bge-large-en-v1.5")
    vectors = e.embed_texts(["hello world", "the quick brown fox"])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 1024)


def test_embedder_normalises_for_cosine():
    e = Embedder("BAAI/bge-large-en-v1.5")
    v = e.embed_texts(["test"])
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedder_prefixes_queries():
    """Queries get a bge-specific instruction prefix; passages do not."""
    e = Embedder("BAAI/bge-large-en-v1.5")
    q = e.embed_query("what is attention")
    p = e.embed_texts(["what is attention"])
    # Same surface text but prefixed query produces a different vector
    assert not np.allclose(q, p[0])
