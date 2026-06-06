from __future__ import annotations

from functools import lru_cache

from anthropic import Anthropic

from app.config import get_settings
from app.rag.generator import Generator
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.stores.keyword_store import KeywordStore
from app.stores.query_log import QueryLog
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


@lru_cache(maxsize=1)
def _reranker() -> Reranker:
    return Reranker(get_settings().reranker_model)


def get_retriever() -> Retriever:
    return Retriever(
        embedder=_embedder(),
        store=_vector_store(),
        keyword_store=_keyword_store(),
        reranker=_reranker(),
    )


def get_generator() -> Generator:
    return Generator(client=_anthropic_client(), model=get_settings().claude_model)


@lru_cache(maxsize=1)
def _query_log() -> QueryLog:
    return QueryLog(get_settings().query_log_db_path)


@lru_cache(maxsize=1)
def _rewriter() -> QueryRewriter:
    return QueryRewriter(client=_anthropic_client(), model=get_settings().rewriter_model)


def get_query_log() -> QueryLog:
    return _query_log()


def get_query_rewriter() -> QueryRewriter:
    return _rewriter()
