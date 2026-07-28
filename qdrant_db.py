"""
Qdrant connection management.

Provides a single reused QdrantClient and a LangChain QdrantVectorStore
wrapper around the configured collection. The collection is created
automatically (with the correct vector size/distance) if it doesn't exist.
"""

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore

from config import get_settings
from embeddings import get_embeddings
from logging_config import get_logger

logger = get_logger()


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Return a cached singleton Qdrant client, reused across requests."""
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def ensure_collection_exists() -> None:
    """Create the configured collection if it does not already exist."""
    settings = get_settings()
    client = get_qdrant_client()

    existing = {c.name for c in client.get_collections().collections}
    if settings.collection_name in existing:
        return

    logger.info("Creating Qdrant collection '%s'", settings.collection_name)
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=qmodels.VectorParams(
            size=settings.embedding_dimension,
            distance=qmodels.Distance.COSINE,
        ),
    )


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    """Return a cached singleton LangChain vector store bound to Qdrant."""
    settings = get_settings()
    ensure_collection_exists()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
    )


def is_connected() -> bool:
    """Lightweight connectivity check used by the /health endpoint."""
    try:
        get_qdrant_client().get_collections()
        return True
    except Exception:  # noqa: BLE001 - health check must never raise
        return False
