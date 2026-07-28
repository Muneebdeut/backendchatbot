"""
FastAPI application entrypoint for the Men's Salon RAG Chatbot.

Endpoints:
    POST /chat     - ask the chatbot a question
    POST /ingest   - (re)ingest every .txt file in data/ into Qdrant
    GET  /health   - liveness / readiness check
    GET  /metrics  - lightweight in-process usage counters
"""

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import get_settings
from guardrails import ADVERTISEMENT_MESSAGE, INJECTION_BLOCKED_MESSAGE, TOXIC_LANGUAGE_MESSAGE
from ingest import run_ingestion
from logging_config import configure_logging, get_logger, instrument_app
from models import ChatRequest, ChatResponse, HealthResponse, IngestResponse, MetricsResponse
from qdrant_db import is_connected
from rag import answer_question

settings = get_settings()
logger = get_logger()
configure_logging()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Men's Salon RAG Chatbot", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

instrument_app(app)

_start_time = time.perf_counter()
_metrics = {
    "total_requests": 0,
    "total_chat_requests": 0,
    "total_guardrail_blocks": 0,
    "total_low_confidence_fallbacks": 0,
    "total_llm_errors": 0,
}


@app.middleware("http")
async def count_requests(request: Request, call_next):
    _metrics["total_requests"] += 1
    return await call_next(request)


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """Answer a user's salon-related question using the RAG pipeline."""
    _metrics["total_chat_requests"] += 1

    answer = await answer_question(payload.message)

    if answer in (ADVERTISEMENT_MESSAGE, INJECTION_BLOCKED_MESSAGE, TOXIC_LANGUAGE_MESSAGE):
        _metrics["total_guardrail_blocks"] += 1

    return ChatResponse(answer=answer)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request) -> IngestResponse:
    """Ingest every .txt file in the data/ directory into Qdrant."""
    logger.info("Ingestion triggered via /ingest endpoint")
    return run_ingestion()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness/readiness check."""
    return HealthResponse(status="ok", qdrant_connected=is_connected())


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Lightweight in-process counters. Detailed traces live in Logfire."""
    return MetricsResponse(
        uptime_seconds=time.perf_counter() - _start_time,
        **_metrics,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
