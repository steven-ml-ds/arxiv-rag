from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.stores.vector_store import VectorStore
from pipeline.chunk import chunk_text
from pipeline.embed import Embedder
from pipeline.fetch_arxiv import fetch_recent_papers
from pipeline.index import index_paper
from pipeline.parse_pdf import pdf_to_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into ChromaDB.")
    parser.add_argument("--max-papers", type=int, default=150)
    parser.add_argument(
        "--categories", nargs="+", default=["cs.LG", "cs.AI", "cs.CL"]
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    settings = get_settings()
    embedder = Embedder(settings.embedding_model)
    store = VectorStore(path=settings.chroma_path, collection=settings.chroma_collection)

    log.info("Fetching up to %d papers from %s", args.max_papers, args.categories)
    papers = fetch_recent_papers(
        categories=args.categories,
        max_papers=args.max_papers,
        pdf_dir=Path(settings.pdf_dir),
    )
    log.info("Downloaded %d PDFs", len(papers))

    indexed = 0
    skipped = 0
    for paper in papers:
        try:
            text = pdf_to_text(paper.pdf_path)
            if not text.strip():
                log.warning("Empty text for %s; skipping", paper.arxiv_id)
                skipped += 1
                continue
            chunks = chunk_text(text, size=args.chunk_size, overlap=args.chunk_overlap)
            embeddings = embedder.embed_texts(chunks).tolist()
            index_paper(paper, chunks, embeddings, store)
            indexed += 1
            log.info("Indexed %s (%d chunks)", paper.arxiv_id, len(chunks))
        except Exception as e:  # noqa: BLE001 — top-level catch is intentional
            log.exception("Failed to index %s: %s", paper.arxiv_id, e)
            skipped += 1

    log.info("Done. Indexed=%d skipped=%d total_in_store=%d", indexed, skipped, store.count())
    return 0 if indexed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
