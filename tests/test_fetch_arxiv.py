from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.fetch_arxiv import Paper, fetch_recent_papers


def test_paper_dataclass_fields():
    p = Paper(
        arxiv_id="2401.12345",
        title="A Test Paper",
        authors=["Alice", "Bob"],
        published_year=2024,
        pdf_path=Path("/tmp/x.pdf"),
    )
    assert p.arxiv_id == "2401.12345"
    assert p.published_year == 2024


def test_fetch_recent_papers_downloads_pdfs(tmp_path):
    """Mocks arxiv.Client and the urllib download; verifies PDFs land in pdf_dir."""
    fake_result = MagicMock()
    fake_result.entry_id = "http://arxiv.org/abs/2401.12345v1"
    fake_result.title = "Test Title"
    fake_result.authors = [MagicMock(), MagicMock()]
    fake_result.authors[0].name = "Alice"
    fake_result.authors[1].name = "Bob"
    fake_result.published.year = 2024

    # arxiv 4.x: PDF URL is on Result.links
    pdf_link = MagicMock()
    pdf_link.title = "pdf"
    pdf_link.content_type = "application/pdf"
    pdf_link.href = "https://arxiv.org/pdf/2401.12345v1"
    fake_result.links = [pdf_link]

    def fake_download(url: str, dest: Path):
        dest.write_bytes(b"%PDF-1.4 stub")

    with patch("pipeline.fetch_arxiv.arxiv.Client") as mock_client, \
         patch("pipeline.fetch_arxiv._download", side_effect=fake_download):
        mock_client.return_value.results.return_value = iter([fake_result])
        papers = fetch_recent_papers(
            categories=["cs.LG"],
            max_papers=1,
            pdf_dir=tmp_path,
        )

    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.12345"
    assert papers[0].title == "Test Title"
    assert papers[0].authors == ["Alice", "Bob"]
    assert papers[0].published_year == 2024
    assert papers[0].pdf_path.exists()
