"""
Embedding model loading via API.

Uses lightweight API-based embeddings (e.g., OpenAI text-embedding-3-small)
to eliminate heavy local PyTorch dependencies for serverless deployment.
"""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from config import get_settings
from logging_config import get_logger

logger = get_logger()


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    """Return a cached singleton embedding model instance."""
    settings = get_settings()
    logger.info("Initializing API embedding model '%s'", settings.embedding_model_name)
    
    kwargs = {"model": settings.embedding_model_name}
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
        
    return OpenAIEmbeddings(**kwargs)

