"""Data models for LLM request logging."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass
class LogEntry:
    """A single LLM request log entry."""

    request_id: UUID
    provider: str
    model: str
    timestamp: datetime
    response_time: float | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    input_cost: float | None
    output_cost: float | None
    total_cost: float | None
    s3_request_key: str | None
    s3_response_key: str | None
    status: str  # "success" or "error"
    error_message: str | None
    api_key_hash: str | None
    api_key_alias: str | None
