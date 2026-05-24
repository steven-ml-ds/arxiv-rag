from __future__ import annotations

from functools import lru_cache

from anthropic import Anthropic
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag.generator import Generator
from app.rag.retriever import Retriever
from app.stores.vector_store import VectorStore
from pipeline.embed import Embedder

router = APIRouter()


class ChatRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=1000)
    top_k: int = 5


class Source(BaseModel):
    arxiv_id: str
    title: str
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


# Heavyweight singletons — built once per process.
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


def get_retriever() -> Retriever:
    return Retriever(embedder=_embedder(), store=_vector_store())


def get_generator() -> Generator:
    return Generator(client=_anthropic_client(), model=get_settings().claude_model)


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
) -> ChatResponse:
    chunks = retriever.retrieve(req.q, top_k=req.top_k)
    answer = generator.generate(req.q, chunks)
    sources = [
        Source(
            arxiv_id=c["metadata"].get("arxiv_id", "?"),
            title=c["metadata"].get("title", "?"),
            chunk_id=c["id"],
        )
        for c in chunks
    ]
    return ChatResponse(answer=answer, sources=sources)
