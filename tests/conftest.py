import numpy as np
import pytest


@pytest.fixture
def fake_vec_1024():
    """Deterministic unit vector for vector-store tests (avoids loading bge)."""
    def _make(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(1024).astype(np.float32)
        return v / np.linalg.norm(v)
    return _make
