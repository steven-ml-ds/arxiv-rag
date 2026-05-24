from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Wraps a sentence-transformers model.

    Use embed_texts(...) for passages (chunks) and embed_query(...) for
    user queries — bge models perform better with an instruction prefix on
    the query side only.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([BGE_QUERY_PREFIX + query])[0]
