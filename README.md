# arxiv-rag

Personal RAG chatbot over arXiv AI/ML papers. Milestone 1: walking skeleton.

## Quick start

```bash
make install
cp .env.example .env  # then fill ANTHROPIC_API_KEY
make pipeline         # ingest 10 papers (~5 min first run, downloads bge model)
make serve            # http://localhost:8000
```

## Test it

```bash
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"q": "What is FlashAttention?"}'
```

## Milestone 2 — better retrieval & answers

M2 upgrades the walking skeleton's retrieval and generation quality:

- **Section-aware chunking** — splits papers by section and drops
  References/bibliography/acknowledgments; each chunk carries its `section`.
- **Hybrid retrieval** — dense ChromaDB vectors + SQLite FTS5 BM25, combined
  with Reciprocal Rank Fusion, then capped per paper for diversity.
- **Reranker** — `BAAI/bge-reranker-base` cross-encoder reranks fused candidates.
- **Structured citations** — answers cite papers inline as `[arxiv:<id>]`, and
  `/chat` returns only the sources actually cited (grounded "I don't know"
  when the corpus lacks coverage).
- **Prompt caching** — the stable system prompt is marked with `cache_control`.

Re-running the pipeline now also builds a keyword index at `data/keywords.db`
alongside the Chroma store:

```bash
make clean && make pipeline   # rebuilds both vector + keyword indexes
```

## Milestone 3 — online UX

M3 makes the chatbot usable as a live web app:

- **Streaming** — `POST /chat/stream` returns Server-Sent Events: a `sources`
  event, then `token` events as Claude generates, then a `done` event with
  token usage.
- **Chat page** — a minimal vanilla-JS UI at `http://localhost:8000/` that
  streams answers token-by-token and supports follow-up questions.
- **Multi-turn** — both `/chat` and `/chat/stream` accept an optional
  `history` array (`[{role, content}, ...]`); the last few turns are included
  in the prompt.
- **Query rewriter** — follow-up questions are rewritten into standalone search
  queries by Claude Haiku (`claude-haiku-4-5`) before retrieval (skipped on the
  first turn).
- **Query log** — every request is logged to `data/query_log.db` (question,
  rewritten query, retrieved doc ids, latency, token usage).

```bash
make serve   # then open http://localhost:8000/
curl -N -X POST localhost:8000/chat/stream -H 'Content-Type: application/json' \
  -d '{"q":"What is FlashAttention?"}'
```
