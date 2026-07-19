"""Generate golden-set candidate questions for human review.

Samples indexed papers from the vector store and asks Claude to draft
multi-hop and unanswerable candidates. Writes eval/candidates.json — it NEVER
touches golden_set.json; a human reviews candidates and promotes the good ones.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

CANDIDATES_PATH = Path("eval/candidates.json")

MULTI_HOP_PROMPT = (
    "You write evaluation questions for a RAG system over arXiv AI/ML papers.\n"
    "Below are excerpts from TWO different papers. Write ONE question that can "
    "only be answered well by combining information from BOTH papers (compare, "
    "contrast, or connect them). The question must not name either paper's "
    "arxiv id and must be natural — something a researcher would actually ask.\n"
    "Reply with strict JSON: {{\"question\": \"...\", \"rationale\": \"why both "
    "papers are required\"}} and nothing else.\n\n"
    "Paper 1 (arxiv_id {id1}, title: {title1}):\n{excerpt1}\n\n"
    "Paper 2 (arxiv_id {id2}, title: {title2}):\n{excerpt2}"
)

UNANSWERABLE_PROMPT = (
    "You write evaluation questions for a RAG system over arXiv AI/ML papers.\n"
    "The corpus contains ONLY the papers listed below. Write {n} questions that "
    "are on-topic for AI/ML research but CANNOT be answered from these papers — "
    "e.g. about methods, systems, or results none of them cover. Avoid "
    "questions whose general background these papers might still answer.\n"
    "Reply with strict JSON: {{\"questions\": [{{\"question\": \"...\", "
    "\"rationale\": \"why it is out of corpus\"}}]}} and nothing else.\n\n"
    "Indexed paper titles:\n{titles}"
)


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.index("{"):]
    return text


def sample_papers(store, max_chunks: int = 2000) -> dict[str, dict[str, str]]:
    """First chunk + title per arxiv_id, keyed by arxiv_id."""
    papers: dict[str, dict[str, str]] = {}
    offset = 0
    while offset < max_chunks:
        batch = store.scan(limit=200, offset=offset)
        if not batch:
            break
        for c in batch:
            aid = c["metadata"].get("arxiv_id")
            if aid and aid not in papers:
                papers[aid] = {
                    "title": c["metadata"].get("title", "?"),
                    "excerpt": c["document"][:1500],
                }
        offset += len(batch)
    return papers


def generate_multi_hop(client, model: str, papers: dict, n: int, rng: random.Random) -> list[dict]:
    ids = list(papers)
    out: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    while len(out) < n and len(seen_pairs) < n * 4:
        id1, id2 = rng.sample(ids, 2)
        pair = tuple(sorted((id1, id2)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        prompt = MULTI_HOP_PROMPT.format(
            id1=id1, title1=papers[id1]["title"], excerpt1=papers[id1]["excerpt"],
            id2=id2, title2=papers[id2]["title"], excerpt2=papers[id2]["excerpt"],
        )
        resp = client.messages.create(
            model=model, max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            data = json.loads(_strip_fence(resp.content[0].text))
            out.append({
                "question": data["question"],
                "expected_arxiv_ids": [id1, id2],
                "category": "multi_hop",
                "rationale": data.get("rationale", ""),
                "titles": [papers[id1]["title"], papers[id2]["title"]],
            })
            print(f"  multi-hop {len(out)}/{n}: {data['question'][:70]}...")
        except (ValueError, KeyError) as err:
            print(f"  skipped malformed multi-hop response ({err})")
    return out


def generate_unanswerable(client, model: str, papers: dict, n: int) -> list[dict]:
    titles = "\n".join(f"- {p['title']}" for p in papers.values())
    prompt = UNANSWERABLE_PROMPT.format(n=n, titles=titles)
    resp = client.messages.create(
        model=model, max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    data = json.loads(_strip_fence(resp.content[0].text))
    return [
        {
            "question": q["question"],
            "expected_arxiv_ids": [],
            "category": "unanswerable",
            "rationale": q.get("rationale", ""),
        }
        for q in data["questions"][:n]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-hop", type=int, default=16,
                        help="multi-hop candidates to draft (default 16, review down to 12)")
    parser.add_argument("--unanswerable", type=int, default=12,
                        help="unanswerable candidates to draft (default 12, review down to 8)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from anthropic import Anthropic

    from app.config import get_settings
    from app.stores.vector_store import VectorStore

    settings = get_settings()
    store = VectorStore(path=settings.chroma_path, collection=settings.chroma_collection)
    client = Anthropic(api_key=settings.anthropic_api_key)

    papers = sample_papers(store)
    if len(papers) < 2:
        print("Vector store has fewer than 2 papers — run the pipeline first.")
        return 1
    print(f"Sampled {len(papers)} papers from the index.")

    rng = random.Random(args.seed)
    candidates = {
        "multi_hop": generate_multi_hop(
            client, settings.claude_model, papers, args.multi_hop, rng
        ),
        "unanswerable": generate_unanswerable(
            client, settings.claude_model, papers, args.unanswerable
        ),
    }

    CANDIDATES_PATH.write_text(json.dumps(candidates, indent=2, ensure_ascii=False))
    print(
        f"\nWrote {CANDIDATES_PATH} "
        f"({len(candidates['multi_hop'])} multi-hop, "
        f"{len(candidates['unanswerable'])} unanswerable).\n"
        "Review them, then merge approved items into eval/golden_set.json."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
