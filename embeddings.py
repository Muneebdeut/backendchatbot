"""
Embedding model loading.

Uses sentence-transformers/all-MiniLM-L6-v2 via HuggingFace, downloaded once
and cached to disk (EMBEDDING_CACHE_DIR) so subsequent startups load from
the local cache instead of re-downloading.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import get_settings
from logging_config import get_logger

logger = get_logger()


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached singleton embedding model instance.

    The lru_cache ensures the (relatively expensive) model load only
    happens once per process, and the cache_folder ensures the weights
    are only downloaded once per machine.
    """
    settings = get_settings()
    logger.info(
        "Loading embedding model '%s' (cache: %s)",
        settings.embedding_model_name,
        settings.embedding_cache_path,
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        cache_folder=str(settings.embedding_cache_path),
        encode_kwargs={"normalize_embeddings": True},
    )
