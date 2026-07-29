"""
Application configuration.

All runtime configuration is loaded from environment variables (via a .env
file in development). Nothing is hardcoded so the same image can be
deployed to any environment just by changing env vars.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings, populated from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Qdrant ---
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "mens_saloon"

    # --- LLM providers (used by LiteLLM Router & Embeddings) ---
    openai_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    mistral_api_key: str = ""

    # --- Observability ---
    logfire_token: str = ""

    # --- Embedding model ---
    embedding_model_name: str = "text-embedding-3-small"
    embedding_cache_dir: str = "./model_cache"
    embedding_dimension: int = 1536

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retrieval (MMR) ---
    retriever_k: int = 5
    retriever_fetch_k: int = 20
    retriever_lambda_mult: float = 0.5

    # Minimum cosine similarity a top match must have before we even
    # bother calling the LLM. Below this, we assume the question is
    # off-topic / not covered by our documents and show the ad instead.
    similarity_threshold: float = 0.35

    # --- Data ---
    data_dir: str = "./data"

    # --- API / security ---
    cors_origins: str = "http://localhost:5173"
    rate_limit_per_minute: int = 20

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def embedding_cache_path(self) -> Path:
        path = Path(self.embedding_cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_path(self) -> Path:
        path = Path(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
