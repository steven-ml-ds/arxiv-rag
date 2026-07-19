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
     Golden set (40 Q) ──▶ hit@5 · citations · refusals · faithfulness
```

## Evaluation (M5: hardened)

The original 20-question golden set saturated (hit@5 = 1.00, citation_accuracy
= 1.00) — it could no longer tell a good change from a bad one. M5 expands it
to **40 questions across three categories**, each targeting a different failure
mode, over 300 indexed arXiv AI/ML papers (June 2026 batch):

| Category | n | Scored by | Catches |
|----------|---|-----------|---------|
| single_hop | 20 | any-hit@5 + citation_accuracy | basic retrieval/citation regressions |
| multi_hop | 12 | **all-hit@5** (both papers required) + citation_accuracy | retrieval that can't cover compound questions |
| unanswerable | 8 | refusal_accuracy (sentinel + zero citations) | hallucinated answers to out-of-corpus questions |

Multi-hop questions are phrased conceptually (no paper-specific vocabulary),
so lexical matching alone can't find the sources. Unanswerable questions are
deliberate traps: on-topic, with near-neighbor papers in the corpus
(e.g. asking about vLLM's PagedAttention while the corpus contains
SegPagedAttention) — the system must notice that *related ≠ sufficient*.

Every run also reports `citation_grounding` (cited ids must actually appear in
the retrieved context — anything else is fabrication) and `false_refusal_rate`
(refusing questions it should answer). `--judge` adds claim-level faithfulness
scoring with `claude-haiku-4-5` as LLM-as-judge.

**Retrieval baseline** (`uv run python eval/run_eval.py --retrieval-only`, no API needed):

| Category | all/any-hit@5 |
|----------|---------------|
| single_hop | 1.00 (20/20) |
| multi_hop | 0.58 (7/12) |

single_hop staying at 1.00 confirms hardening didn't break the base case;
multi_hop at 0.58 is the new headroom to optimize against. Generation-side
scores (citations, refusals, faithfulness) come from the full run:
`uv run python eval/run_eval.py [--judge]`, written to `data/eval_results.json`
and surfaced at `/dashboard`.

New questions are drafted with `eval/generate_questions.py` (samples the
indexed corpus, proposes multi-hop pairs and unanswerable traps into
`eval/candidates.json`) and **hand-reviewed** before entering the golden set.

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

- **Golden set** — `eval/golden_set.json`: hand-reviewed questions with
  expected arxiv_ids (grown to 40 across three categories in M5; see
  Evaluation above). Edit the ids to match the papers you actually indexed.
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
