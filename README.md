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

See `/Users/yun/.claude/plans/rag-chatbot-user-ask-concurrent-scroll.md` for the master plan
and `/Users/yun/.claude/plans/rag-chatbot-m1-walking-skeleton.md` for M1 details.
