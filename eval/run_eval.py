from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from eval.metrics import (
    all_hit_at_k,
    citation_accuracy,
    citation_grounding,
    hit_at_k,
    is_refusal,
)

ANSWERABLE = ("single_hop", "multi_hop")


def _score_row(item: dict, retrieved_ids: list[str], answer: str, top_k: int) -> dict[str, Any]:
    """Score one golden-set item according to its category."""
    category = item.get("category", "single_hop")
    expected = item["expected_arxiv_ids"]
    row: dict[str, Any] = {
        "question": item["question"],
        "category": category,
        "citation_grounding": citation_grounding(answer, retrieved_ids),
        "refused": is_refusal(answer),
    }
    if category == "unanswerable":
        # Correct behavior: refuse, and cite nothing while doing so.
        row["refusal_correct"] = row["refused"] and citation_grounding(answer, []) == 1.0
    else:
        hit_fn = all_hit_at_k if category == "multi_hop" else hit_at_k
        row["hit"] = hit_fn(retrieved_ids, expected, top_k)
        row["citation_accuracy"] = citation_accuracy(answer, expected)
    return row


def _score_retrieval_row(item: dict, retrieved_ids: list[str], top_k: int) -> dict[str, Any]:
    """Score only the retrieval half of one item (no answer available)."""
    category = item.get("category", "single_hop")
    row: dict[str, Any] = {"question": item["question"], "category": category}
    if category != "unanswerable":
        hit_fn = all_hit_at_k if category == "multi_hop" else hit_at_k
        row["hit"] = hit_fn(retrieved_ids, item["expected_arxiv_ids"], top_k)
    return row


def _mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return sum(vals) / len(vals) if vals else 0.0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["category"] in ANSWERABLE]
    by_category: dict[str, dict[str, Any]] = {}
    for cat in ("single_hop", "multi_hop", "unanswerable"):
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        stats: dict[str, Any] = {"n": len(cat_rows)}
        # Only report metrics some row actually produced — retrieval-only runs
        # have no generation metrics, and a fake 0.00 would read as a failure.
        sources = {"refusal_accuracy": "refusal_correct", "hit_at_5": "hit",
                   "citation_accuracy": "citation_accuracy",
                   "citation_grounding": "citation_grounding",
                   "faithfulness": "faithfulness"}
        for out_key, row_key in sources.items():
            if any(row_key in r for r in cat_rows):
                stats[out_key] = _mean(cat_rows, row_key)
        by_category[cat] = stats

    return {
        "n": len(rows),
        # Top-level keys kept for the dashboard; computed over answerable rows
        # so refusals don't drag averages that don't apply to them.
        "hit_at_5": _mean(answerable, "hit"),
        "citation_accuracy": _mean(answerable, "citation_accuracy"),
        "avg_latency_ms": _mean(rows, "latency_ms"),
        "citation_grounding": _mean(rows, "citation_grounding"),
        "false_refusal_rate": _mean(answerable, "refused"),
        "by_category": by_category,
        "rows": rows,
    }


def evaluate(
    golden: list[dict],
    retriever,
    generator=None,
    top_k: int = 5,
    judge=None,
) -> dict[str, Any]:
    """Run the golden set through retrieval + generation, aggregating metrics.

    `generator=None` runs retrieval-only: hit metrics are computed, generation
    metrics (citations, refusals, faithfulness) are skipped — useful when the
    retrieval stack is available but the LLM API is not.
    `judge` is an optional callable (question, chunks, answer) -> dict merged
    into the row (see eval.judge.make_judge).
    """
    rows: list[dict[str, Any]] = []
    for item in golden:
        question = item["question"]
        started = time.monotonic()
        chunks = retriever.retrieve(question, top_k=top_k)
        retrieved_ids = [c["metadata"].get("arxiv_id", "?") for c in chunks]
        if generator is None:
            row = _score_retrieval_row(item, retrieved_ids, top_k)
        else:
            answer = generator.generate(question, chunks)
            row = _score_row(item, retrieved_ids, answer, top_k)
            if judge is not None:
                row.update(judge(question, chunks, answer))
        row["latency_ms"] = (time.monotonic() - started) * 1000
        rows.append(row)
    return _aggregate(rows)


def _print_report(results: dict[str, Any], retrieval_only: bool = False) -> None:
    print(f"n:                  {results['n']}")
    print(f"hit@5 (answerable): {results['hit_at_5']:.2f}")
    if retrieval_only:
        print("(retrieval-only run: generation metrics skipped)")
    else:
        print(f"citation_accuracy:  {results['citation_accuracy']:.2f}")
        print(f"citation_grounding: {results['citation_grounding']:.2f}")
        print(f"false_refusal_rate: {results['false_refusal_rate']:.2f}")
    print(f"avg_latency_ms:     {results['avg_latency_ms']:.0f}")
    for cat, stats in results["by_category"].items():
        parts = [f"n={stats['n']}"]
        for key in ("hit_at_5", "citation_accuracy", "refusal_accuracy",
                    "citation_grounding", "faithfulness"):
            if key in stats:
                parts.append(f"{key}={stats[key]:.2f}")
        print(f"  {cat:12s} {' '.join(parts)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the golden-set RAG eval.")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="also run LLM-as-judge faithfulness scoring (extra API calls)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip generation entirely (no API calls); hit metrics only",
    )
    args = parser.parse_args()
    if args.judge and args.retrieval_only:
        parser.error("--judge requires generation; drop --retrieval-only")

    from app.api.deps import get_generator, get_retriever
    from app.config import get_settings

    settings = get_settings()
    golden = json.loads(Path(settings.golden_set_path).read_text())

    judge = None
    if args.judge:
        from anthropic import Anthropic

        from eval.judge import make_judge

        judge = make_judge(Anthropic(api_key=settings.anthropic_api_key))

    generator = None if args.retrieval_only else get_generator()
    results = evaluate(golden, get_retriever(), generator, judge=judge)
    _print_report(results, retrieval_only=args.retrieval_only)

    out = Path(settings.eval_results_path)
    if args.retrieval_only:
        # Don't clobber the canonical results the dashboard reads.
        out = out.with_name(out.stem + "_retrieval_only" + out.suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
