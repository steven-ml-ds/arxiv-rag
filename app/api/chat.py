from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import (
    get_generator,
    get_query_log,
    get_query_rewriter,
    get_retriever,
)
from app.rag.generator import Generator
from app.rag.query_rewriter import QueryRewriter
from app.rag.retriever import Retriever
from app.stores.query_log import QueryLog

router = APIRouter()


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=1000)
    top_k: int = 5
    history: list[Turn] = Field(default_factory=list)


class Source(BaseModel):
    arxiv_id: str
    title: str
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


def _all_sources(chunks: list[dict]) -> list[Source]:
    out: list[Source] = []
    seen: set[str] = set()
    for c in chunks:
        aid = c["metadata"].get("arxiv_id", "?")
        if aid in seen:
            continue
        seen.add(aid)
        out.append(
            Source(arxiv_id=aid, title=c["metadata"].get("title", "?"), chunk_id=c["id"])
        )
    return out


def cited_sources(answer: str, chunks: list[dict]) -> list[Source]:
    """Sources actually cited in the answer ([arxiv:<id>]); fall back to all."""
    alls = _all_sources(chunks)
    cited = [s for s in alls if f"[arxiv:{s.arxiv_id}]" in answer]
    return cited if cited else alls


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
    rewriter: QueryRewriter = Depends(get_query_rewriter),
    query_log: QueryLog = Depends(get_query_log),
) -> ChatResponse:
    history = [t.model_dump() for t in req.history]
    started = time.monotonic()
    search_q = rewriter.rewrite(req.q, history)
    chunks = retriever.retrieve(search_q, top_k=req.top_k)
    answer = generator.generate(req.q, chunks, history=history)
    query_log.log(
        req.q, search_q, [c["id"] for c in chunks],
        (time.monotonic() - started) * 1000, None, None,
    )
    return ChatResponse(answer=answer, sources=cited_sources(answer, chunks))


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat/stream")
def chat_stream(
    req: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
    rewriter: QueryRewriter = Depends(get_query_rewriter),
    query_log: QueryLog = Depends(get_query_log),
) -> StreamingResponse:
    history = [t.model_dump() for t in req.history]
    started = time.monotonic()
    search_q = rewriter.rewrite(req.q, history)
    chunks = retriever.retrieve(search_q, top_k=req.top_k)

    def gen() -> Iterator[str]:
        yield _sse("sources", {"sources": [s.model_dump() for s in _all_sources(chunks)]})
        in_tok = out_tok = None
        for ev in generator.generate_stream(req.q, chunks, history=history):
            if ev["type"] == "text":
                yield _sse("token", {"text": ev["text"]})
            elif ev["type"] == "usage":
                in_tok, out_tok = ev["input_tokens"], ev["output_tokens"]
        query_log.log(
            req.q, search_q, [c["id"] for c in chunks],
            (time.monotonic() - started) * 1000, in_tok, out_tok,
        )
        yield _sse("done", {"input_tokens": in_tok, "output_tokens": out_tok})

    return StreamingResponse(gen(), media_type="text/event-stream")
