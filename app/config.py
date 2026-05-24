from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    chroma_path: str = "data/chroma"
    chroma_collection: str = "papers"
    pdf_dir: str = "data/pdfs"


def get_settings() -> Settings:
    """Factory so tests can override and the app can cache."""
    return Settings()
