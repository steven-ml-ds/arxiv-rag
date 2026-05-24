from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arxiv


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    published_year: int
    pdf_path: Path


def _arxiv_id_from_entry_id(entry_id: str) -> str:
    """Convert 'http://arxiv.org/abs/2401.12345v1' → '2401.12345'."""
    raw = entry_id.rsplit("/", 1)[-1]
    return raw.split("v")[0]


def fetch_recent_papers(
    categories: list[str],
    max_papers: int,
    pdf_dir: Path,
) -> list[Paper]:
    """Fetch the latest N papers from the given arXiv categories.

    Downloads PDFs into pdf_dir/<arxiv_id>.pdf and returns Paper records.
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    query = " OR ".join(f"cat:{c}" for c in categories)

    search = arxiv.Search(
        query=query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)

    papers: list[Paper] = []
    for result in client.results(search):
        arxiv_id = _arxiv_id_from_entry_id(result.entry_id)
        filename = f"{arxiv_id}.pdf"
        target = pdf_dir / filename
        if not target.exists():
            result.download_pdf(dirpath=str(pdf_dir), filename=filename)
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=result.title.strip(),
                authors=[a.name for a in result.authors],
                published_year=result.published.year,
                pdf_path=target,
            )
        )
    return papers
