from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from eval.metrics import citation_accuracy, hit_at_k


def evaluate(golden: list[dict], retriever, generator, top_k: int = 5) -> dict[str, Any]:
    """Run the golden set through retrieval + generation, aggregating metrics."""
    rows: list[dict[str, Any]] = []
    for item in golden:
        question = item["question"]
        expected = item["expected_arxiv_ids"]
        started = time.monotonic()
        chunks = retriever.retrieve(question, top_k=top_k)
        retrieved_ids = [c["metadata"].get("arxiv_id", "?") for c in chunks]
        answer = generator.generate(question, chunks)
        latency_ms = (time.monotonic() - started) * 1000
        rows.append({
            "question": question,
            "hit": hit_at_k(retrieved_ids, expected, top_k),
            "citation_accuracy": citation_accuracy(answer, expected),
            "latency_ms": latency_ms,
        })
    n = len(rows)
    return {
        "n": n,
        "hit_at_5": sum(r["hit"] for r in rows) / n if n else 0.0,
        "citation_accuracy": sum(r["citation_accuracy"] for r in rows) / n if n else 0.0,
        "avg_latency_ms": sum(r["latency_ms"] for r in rows) / n if n else 0.0,
        "rows": rows,
    }


def main() -> int:
    from app.api.deps import get_generator, get_retriever
    from app.config import get_settings

    settings = get_settings()
    golden = json.loads(Path(settings.golden_set_path).read_text())
    results = evaluate(golden, get_retriever(), get_generator())

    print(f"hit@5:             {results['hit_at_5']:.2f}")
    print(f"citation_accuracy: {results['citation_accuracy']:.2f}")
    print(f"avg_latency_ms:    {results['avg_latency_ms']:.0f}")
    print(f"n:                 {results['n']}")

    out = Path(settings.eval_results_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
