from __future__ import annotations

from functools import lru_cache

from anthropic import Anthropic

from app.config import get_settings
from app.rag.generator import Generator
from app.rag.retriever import Retriever
from app.stores.keyword_store import KeywordStore
from app.stores.vector_store import VectorStore
from pipeline.embed import Embedder


# Heavyweight singletons — built once per process and shared across routers.
@lru_cache(maxsize=1)
def _embedder() -> Embedder:
    return Embedder(get_settings().embedding_model)


@lru_cache(maxsize=1)
def _vector_store() -> VectorStore:
    s = get_settings()
    return VectorStore(path=s.chroma_path, collection=s.chroma_collection)


@lru_cache(maxsize=1)
def _anthropic_client() -> Anthropic:
    return Anthropic(api_key=get_settings().anthropic_api_key)


@lru_cache(maxsize=1)
def _keyword_store() -> KeywordStore:
    return KeywordStore(get_settings().keyword_db_path)


def get_retriever() -> Retriever:
    return Retriever(
        embedder=_embedder(),
        store=_vector_store(),
        keyword_store=_keyword_store(),
    )


def get_generator() -> Generator:
    return Generator(client=_anthropic_client(), model=get_settings().claude_model)
