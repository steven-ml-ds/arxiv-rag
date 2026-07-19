# arxiv-rag

Personal RAG chatbot over arXiv AI/ML papers with hybrid retrieval, reranking, streaming UX, and evaluation.

## Quick start

```bash
make install
cp .env.example .env  # fill ANTHROPIC_API_KEY
make pipeline         # ingest papers (~5 min first run, downloads bge model)
make serve            # http://localhost:8000
```

## Demo

**Q:** What is FlashAttention?

**A (with citations):**
> FlashAttention [arxiv:2205.14135] is an IO-aware exact attention algorithm that tiles
> attention computation into blocks that fit in GPU SRAM, dramatically reducing HBM
> reads/writes. It achieves 2–4× speedup over standard attention while producing
> numerically identical results, enabling training of longer sequences without memory blow-up.

## Architecture

```
PDFs (arXiv papers)
     │
     ▼  pipeline/ingest.py
     │  • section-aware chunking — drops References/Acknowledgments
     │  • embed: BAAI/bge-large-en-v1.5
     ├──▶ ChromaDB        (dense vector index)
     └──▶ SQLite FTS5     (BM25 keyword index)
                │
                ▼  pipeline/retrieve.py
         Hybrid retrieval
          dense ──┐
          BM25  ──┼──▶ RRF fusion ──▶ BAAI/bge-reranker-base ──▶ top-k chunks
                  │
                  ▼  app/rag/
           Claude (claude-sonnet-4-6)
            • system-prompt caching (cache_control)
            • structured inline citations [arxiv:<id>]
            • query rewriting for follow-ups (claude-haiku-4-5)
            • streaming SSE (token-by-token)
                  │
                  ▼
     POST /chat  ·  POST /chat/stream
                  │
                  ▼  eval/run_eval.py
     Golden set (20 Q) ──▶ hit@5 · citation_accuracy · avg_latency_ms
```

## Retrieval Evaluation

20-question golden set (`eval/golden_set.json`) measures end-to-end RAG quality.
Run `uv run python eval/run_eval.py` (requires `ANTHROPIC_API_KEY` + indexed corpus);
results are written to `data/eval_results.json` and surfaced at `/dashboard`.

| Metric | Definition | Score |
|--------|-----------|-------|
| hit@5 (retrieval) | ≥ 1 expected arxiv_id in top-5 retrieved chunks | **1.00** (20/20) |
| citation_accuracy (generation) | Expected ids cited in the answer | **1.00** (20/20) |
| avg_latency_ms | End-to-end per question | ~9 000 ms |

Evaluated on 20 hand-written questions over 150 indexed arXiv AI/ML papers (June 2026 batch).
Run `uv run python eval/run_eval.py` to reproduce; results are written to `data/eval_results.json`.

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

## Milestone 4 — evaluation & scheduling

M4 adds RAG evaluation discipline and scheduled ingestion:

- **Golden set** — `eval/golden_set.json`: 20 hand-written questions with
  expected arxiv_ids. Edit the ids to match the papers you actually indexed.
- **Eval** — `uv run python eval/run_eval.py` runs the golden set through retrieval +
  generation and reports `hit@5`, `citation_accuracy`, and `avg_latency_ms`,
  writing `data/eval_results.json`.
- **Dashboard** — `http://localhost:8000/dashboard` shows the latest eval
  metrics and recent query-log rows (served from `GET /api/stats`).
- **Weekly ingestion (Airflow)** — `airflow/dags/arxiv_ingest_dag.py` runs the
  pipeline every Monday 03:00 with retries.

```bash
# Evaluate (needs ANTHROPIC_API_KEY + an indexed corpus)
uv run python eval/run_eval.py

# Dashboard
make serve   # then open http://localhost:8000/dashboard

# Airflow (optional)
uv pip install -e ".[airflow]"
AIRFLOW__CORE__DAGS_FOLDER="$PWD/airflow/dags" airflow standalone
```
