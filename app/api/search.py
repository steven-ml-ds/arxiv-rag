from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_retriever
from app.rag.retriever import Retriever

router = APIRouter()

MAX_MATCHED_CHUNKS = 3
SNIPPET_LENGTH = 300


class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)


class MatchedChunk(BaseModel):
    chunk_id: str
    chunk_index: int
    snippet: str
    distance: float


class ArticleResult(BaseModel):
    arxiv_id: str
    title: str
    authors: str
    year: int | None
    score: float
    matched_chunks: list[MatchedChunk]


class SearchResponse(BaseModel):
    results: list[ArticleResult]


def _snippet(document: str) -> str:
    text = " ".join(document.split())
    if len(text) <= SNIPPET_LENGTH:
        return text
    return text[:SNIPPET_LENGTH].rstrip() + "..."


def _score(distance: float) -> float:
    return round(max(0.0, 1.0 - distance), 4)


def group_chunks_by_article(
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[ArticleResult]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        arxiv_id = metadata.get("arxiv_id") or "unknown"
        grouped.setdefault(arxiv_id, []).append(chunk)

    results: list[ArticleResult] = []
    for arxiv_id, article_chunks in grouped.items():
        ranked_chunks = sorted(article_chunks, key=lambda c: c.get("distance", 1.0))
        best = ranked_chunks[0]
        metadata = best.get("metadata") or {}
        best_distance = float(best.get("distance", 1.0))
        matched_chunks = [
            MatchedChunk(
                chunk_id=str(chunk.get("id", "")),
                chunk_index=int((chunk.get("metadata") or {}).get("chunk_index") or 0),
                snippet=_snippet(str(chunk.get("document", ""))),
                distance=float(chunk.get("distance", 1.0)),
            )
            for chunk in ranked_chunks[:MAX_MATCHED_CHUNKS]
        ]
        results.append(
            ArticleResult(
                arxiv_id=str(arxiv_id),
                title=str(metadata.get("title") or "Unknown title"),
                authors=str(metadata.get("authors") or ""),
                year=metadata.get("year"),
                score=_score(best_distance),
                matched_chunks=matched_chunks,
            )
        )

    return sorted(
        results,
        key=lambda result: result.matched_chunks[0].distance,
    )[:top_k]


@router.post("/search", response_model=SearchResponse)
def search(
    req: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    # /search does its own per-article grouping and shows up to
    # MAX_MATCHED_CHUNKS snippets per paper, so it must allow that many chunks
    # per arxiv_id through the retriever's diversity cap (whose default of 2 is
    # tuned for /chat, not search).
    chunks = retriever.retrieve(
        req.q, top_k=req.top_k * 3, max_per_paper=MAX_MATCHED_CHUNKS
    )
    return SearchResponse(results=group_chunks_by_article(chunks, top_k=req.top_k))
