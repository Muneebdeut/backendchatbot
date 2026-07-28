"""
Observability setup.

Wires up Logfire for structured logging + tracing across the FastAPI app,
the RAG pipeline, Qdrant retrieval, and LiteLLM calls. Falls back to plain
structured console logging if no LOGFIRE_TOKEN is configured, so the app
still runs (and logs usefully) without an observability backend.
"""

import logging

import logfire

from config import get_settings

logger = logging.getLogger("mens_salon_chatbot")


def configure_logging() -> None:
    """Configure Logfire + standard logging. Call once at startup."""
    settings = get_settings()

    logfire.configure(
        token=settings.logfire_token or None,
        send_to_logfire="if-token-present",
        service_name="mens-salon-chatbot",
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.setLevel(logging.INFO)

    # Route standard library logs (uvicorn, etc.) through Logfire as well.
    logfire.instrument_pydantic()


def instrument_app(app) -> None:
    """Instrument the FastAPI app instance for request/response tracing."""
    #logfire.instrument_fastapi(app, capture_headers=False)
    # pass


def get_logger() -> logging.Logger:
    return logger
