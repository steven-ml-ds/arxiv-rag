import os
from app.config import Settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    monkeypatch.setenv("CHROMA_PATH", "data/chroma")
    monkeypatch.setenv("CHROMA_COLLECTION", "papers")
    monkeypatch.setenv("PDF_DIR", "data/pdfs")

    s = Settings()
    assert s.anthropic_api_key == "sk-test"
    assert s.claude_model == "claude-sonnet-4-6"
    assert s.embedding_model == "BAAI/bge-large-en-v1.5"
    assert s.chroma_path == "data/chroma"
    assert s.chroma_collection == "papers"
    assert s.pdf_dir == "data/pdfs"


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # Clear non-required envs to confirm defaults
    for k in ["CLAUDE_MODEL", "EMBEDDING_MODEL", "CHROMA_PATH",
              "CHROMA_COLLECTION", "PDF_DIR"]:
        monkeypatch.delenv(k, raising=False)
    s = Settings()
    assert s.claude_model == "claude-sonnet-4-6"
    assert s.chroma_collection == "papers"
