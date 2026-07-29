"""
Embedding model loading via HTTP API.

Uses OpenAIEmbeddings if OPENAI_API_KEY is provided, or falls back to a lightweight
HuggingFaceAPIEmbeddings class querying Hugging Face Inference API via HTTP (zero PyTorch footprint).
"""

import os
from functools import lru_cache

import requests
from langchain_core.embeddings import Embeddings

from config import get_settings
from logging_config import get_logger

logger = get_logger()


class HuggingFaceAPIEmbeddings(Embeddings):
    """Lightweight Embeddings class using Hugging Face Inference API via HTTP."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", api_key: str = ""):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=30,
        )
        response.raise_for_status()
        res = response.json()
        # If single text returned 2D array, unwrap if needed
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], float):
            return [res]
        return res

    def embed_query(self, text: str) -> list[float]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30,
        )
        response.raise_for_status()
        res = response.json()
        if isinstance(res, list) and len(res) > 0:
            if isinstance(res[0], list):
                return res[0]
            if isinstance(res[0], float):
                return res
        return res


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

    # Fallback to Hugging Face HTTP Inference API (free, zero PyTorch footprint)
    hf_model = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info("OPENAI_API_KEY not set. Using Hugging Face HTTP Inference API for '%s'", hf_model)
    hf_token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN", "")

    return HuggingFaceAPIEmbeddings(model_name=hf_model, api_key=hf_token)



