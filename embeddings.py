"""
Embedding model loading via HTTP API.

Supports:
1. OpenAIEmbeddings if OPENAI_API_KEY is configured.
2. Hugging Face Router API (https://router.huggingface.co/hf-inference/v1) if HUGGINGFACE_API_KEY / HF_TOKEN is configured.
"""

import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from config import get_settings
from logging_config import get_logger

logger = get_logger()


@lru_cache
def get_embeddings() -> Embeddings:
    """Return a cached singleton embedding model instance."""
    settings = get_settings()

    openai_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        logger.info("Initializing OpenAI API embedding model '%s'", settings.embedding_model_name)
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model_name,
            dimensions=settings.embedding_dimension,
            api_key=openai_key,
        )

    hf_token = (
        os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HUGGINGFACE_API_KEY")
        or os.environ.get("HF_TOKEN", "")
    )
    if hf_token:
        hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("Initializing Hugging Face Router embedding model '%s'", hf_model)
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=hf_model,
            base_url="https://router.huggingface.co/hf-inference/v1",
            api_key=hf_token,
            check_embedding_ctx_length=False,
        )

    raise ValueError(
        "No embedding API key found. Please set OPENAI_API_KEY or HUGGINGFACE_API_KEY (HF_TOKEN) "
        "in your Vercel Environment Variables."
    )




