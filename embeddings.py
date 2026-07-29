"""
Embedding model loading via HTTP API.

Uses OpenAIEmbeddings if OPENAI_API_KEY is provided, or falls back to
HuggingFaceInferenceEmbeddings (free HTTP API) so no local PyTorch or heavy model
weights are installed in serverless functions.
"""

import os
from functools import lru_cache

from config import get_settings
from logging_config import get_logger

logger = get_logger()


@lru_cache
def get_embeddings():
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

    # Fallback to Hugging Face HTTP Inference API (free, zero PyTorch footprint)
    hf_model = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info("OPENAI_API_KEY not set. Using Hugging Face HTTP Inference API model '%s'", hf_model)
    from langchain_community.embeddings import HuggingFaceInferenceEmbeddings

    hf_token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN", "")
    kwargs = {"model_name": hf_model}
    if hf_token:
        kwargs["api_key"] = hf_token

    return HuggingFaceInferenceEmbeddings(**kwargs)


