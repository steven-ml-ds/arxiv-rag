from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_generator, get_retriever
from app.rag.generator import Generator
from app.rag.retriever import Retriever

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
