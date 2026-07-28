"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message from the widget."""

    message: str = Field(..., min_length=1, max_length=200, description="The user's message.")


class ChatResponse(BaseModel):
    """Chatbot answer returned to the widget."""

    answer: str


class IngestResponse(BaseModel):
    """Result summary of an ingestion run."""

    files_processed: int
    chunks_created: int
    chunks_inserted: int
    chunks_skipped_duplicate: int


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool


class MetricsResponse(BaseModel):
    total_requests: int
    total_chat_requests: int
    total_guardrail_blocks: int
    total_low_confidence_fallbacks: int
    total_llm_errors: int
    uptime_seconds: float
