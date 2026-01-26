"""Abstract interfaces for logging adapters."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from majordomo_llm.logging.models import LogEntry


class DatabaseAdapter(ABC):
    """Abstract interface for database logging adapters."""

    @abstractmethod
    async def insert(self, entry: LogEntry) -> None:
        """Insert a log entry."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the database connection."""
        ...


class StorageAdapter(ABC):
    """Abstract interface for blob storage adapters."""

    @abstractmethod
    async def upload(
        self,
        request_id: UUID,
        request_body: dict[str, Any],
        response_content: str | dict[str, Any] | None,
    ) -> tuple[str, str | None]:
        """Upload request and response bodies.

        Args:
            request_id: Unique request identifier.
            request_body: The request parameters.
            response_content: The response content (if successful).

        Returns:
            Tuple of (request_key, response_key). response_key may be None if
            there was no response content.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the storage client."""
        ...
