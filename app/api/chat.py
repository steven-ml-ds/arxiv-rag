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


def cited_sources(answer: str, chunks: list[dict]) -> list[Source]:
    """Return de-duped sources actually cited in the answer ([arxiv:<id>]).

    Falls back to all retrieved papers if the model emitted no citation markers.
    """
    out: list[Source] = []
    seen: set[str] = set()
    for c in chunks:
        aid = c["metadata"].get("arxiv_id", "?")
        if aid in seen:
            continue
        seen.add(aid)
        out.append(
            Source(
                arxiv_id=aid,
                title=c["metadata"].get("title", "?"),
                chunk_id=c["id"],
            )
        )
    cited = [s for s in out if f"[arxiv:{s.arxiv_id}]" in answer]
    return cited if cited else out


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
) -> ChatResponse:
    chunks = retriever.retrieve(req.q, top_k=req.top_k)
    answer = generator.generate(req.q, chunks)
    return ChatResponse(answer=answer, sources=cited_sources(answer, chunks))
