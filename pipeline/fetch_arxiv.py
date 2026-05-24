from __future__ import annotations

import urllib.request
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


def _pdf_url(result) -> str | None:
    """arxiv 4.x: dig the PDF URL out of result.links."""
    for link in result.links:
        if link.title == "pdf" or (link.content_type == "application/pdf"):
            return link.href
    return None


def _download(url: str, dest: Path) -> None:
    """Stream-download a URL to disk."""
    req = urllib.request.Request(url, headers={"User-Agent": "arxiv-rag/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)


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
        target = pdf_dir / f"{arxiv_id}.pdf"
        if not target.exists():
            url = _pdf_url(result)
            if url is None:
                continue  # paper has no PDF link; skip
            _download(url, target)
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
