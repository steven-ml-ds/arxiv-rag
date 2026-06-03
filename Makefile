.PHONY: install test pipeline serve clean

install:
	uv venv && uv pip install -e ".[dev]"

test:
	uv run pytest

pipeline:
	uv run python -m pipeline.run_pipeline --max-papers 150

serve:
	uv run uvicorn app.main:app --reload --port 8000

clean:
	rm -rf data/pdfs/* data/chroma/* .pytest_cache .coverage
