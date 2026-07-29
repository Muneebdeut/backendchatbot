"""
Embedding model loading via HTTP API.

Supports:
1. OpenAIEmbeddings if OPENAI_API_KEY is configured.
2. HuggingFaceRouterEmbeddings querying Hugging Face Router API (https://router.huggingface.co/hf-inference/models)
   if HUGGINGFACEHUB_API_TOKEN / HUGGINGFACE_API_KEY / HF_TOKEN is configured.
"""

import os
from functools import lru_cache

import requests
from langchain_core.embeddings import Embeddings

from config import get_settings
from logging_config import get_logger

logger = get_logger()


class HuggingFaceRouterEmbeddings(Embeddings):
    """Lightweight Embeddings class using Hugging Face Router API via HTTP."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", api_key: str = ""):
        self.model_name = model_name
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def embed_query(self, text: str) -> list[float]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": [text], "options": {"wait_for_model": True}},
            timeout=30,
        )
        response.raise_for_status()
        res = response.json()
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
            return res[0]
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

    hf_token = (
        os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HUGGINGFACE_API_KEY")
        or os.environ.get("HF_TOKEN", "")
    )
    if hf_token:
        hf_model = "BAAI/bge-small-en-v1.5"
        logger.info("Initializing Hugging Face Router embedding model '%s'", hf_model)
        return HuggingFaceRouterEmbeddings(model_name=hf_model, api_key=hf_token)

    raise ValueError(
        "No embedding API key found. Please set OPENAI_API_KEY or HUGGINGFACEHUB_API_TOKEN (HF_TOKEN) "
        "in your Vercel Environment Variables."
    )





