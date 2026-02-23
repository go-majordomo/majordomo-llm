"""Google Cloud Storage adapter."""

import json
from typing import Any
from uuid import UUID

from gcloud.aio.storage import Storage

from majordomo_llm.logging.interfaces import StorageAdapter


class GCSAdapter(StorageAdapter):
    """Google Cloud Storage adapter for storing request/response bodies."""

    def __init__(
        self,
        client: Storage,
        bucket: str,
        prefix: str = "llm-logs",
    ) -> None:
        self._client = client
        self._bucket = bucket
        self._prefix = prefix

    @classmethod
    async def create(
        cls,
        bucket: str,
        prefix: str = "llm-logs",
        project: str | None = None,
        service_account_path: str | None = None,
    ) -> "GCSAdapter":
        """Create a new GCSAdapter.

        Args:
            bucket: GCS bucket name.
            prefix: Key prefix for stored objects.
            project: GCP project ID (optional, uses ADC default if not set).
            service_account_path: Path to service account JSON file (optional).
        """
        client = Storage(service_file=service_account_path)
        return cls(client, bucket, prefix)

    async def upload(
        self,
        request_id: UUID,
        request_body: dict[str, Any],
        response_content: str | dict[str, Any] | None,
    ) -> tuple[str, str | None]:
        """Upload request and response bodies to GCS."""
        request_key = f"{self._prefix}/{request_id}/request.json"
        response_key = f"{self._prefix}/{request_id}/response.json" if response_content else None

        await self._client.upload(
            self._bucket,
            request_key,
            json.dumps(request_body, default=str),
            headers={"Content-Type": "application/json"},
        )

        if response_content is not None:
            body = (
                json.dumps(response_content, default=str)
                if isinstance(response_content, dict)
                else response_content
            )
            await self._client.upload(
                self._bucket,
                response_key,
                body,
                headers={"Content-Type": "application/json"},
            )

        return request_key, response_key

    async def close(self) -> None:
        """Close the GCS client."""
        await self._client.close()
