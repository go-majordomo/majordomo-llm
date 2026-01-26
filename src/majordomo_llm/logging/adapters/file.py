"""Local file storage adapter."""

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import aiofiles

from majordomo_llm.logging.interfaces import StorageAdapter


class FileStorageAdapter(StorageAdapter):
    """Local file system adapter for storing request/response bodies.

    Stores each request and response as separate JSON files in a directory.
    Useful for local development, debugging, and examples where S3 is overkill.

    Files are stored as:
        {base_path}/{request_id}_request.json
        {base_path}/{request_id}_response.json

    Example:
        >>> storage = await FileStorageAdapter.create("./llm_logs")
        >>> logged_llm = LoggingLLM(llm, db, storage)
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    @classmethod
    async def create(cls, base_path: str | Path) -> "FileStorageAdapter":
        """Create a new FileStorageAdapter.

        Args:
            base_path: Directory where log files will be stored.
                Created automatically if it doesn't exist.

        Returns:
            A configured FileStorageAdapter instance.
        """
        path = Path(base_path)
        path.mkdir(parents=True, exist_ok=True)
        return cls(path)

    async def upload(
        self,
        request_id: UUID,
        request_body: dict[str, Any],
        response_content: str | dict[str, Any] | None,
    ) -> tuple[str, str | None]:
        """Store request and response bodies as local JSON files."""
        request_key = f"{request_id}_request.json"
        request_path = self._base_path / request_key

        async with aiofiles.open(request_path, "w") as f:
            await f.write(json.dumps(request_body, indent=2, default=str))

        response_key: str | None = None
        if response_content is not None:
            response_key = f"{request_id}_response.json"
            response_path = self._base_path / response_key

            body = (
                json.dumps(response_content, indent=2, default=str)
                if isinstance(response_content, dict)
                else response_content
            )
            async with aiofiles.open(response_path, "w") as f:
                await f.write(body)

        return request_key, response_key

    async def close(self) -> None:
        """Close the storage adapter (no-op for file storage)."""
        pass
