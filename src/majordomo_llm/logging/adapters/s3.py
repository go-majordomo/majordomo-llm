"""S3 storage adapter."""

import json
from typing import Any
from uuid import UUID

import aioboto3

from majordomo_llm.logging.interfaces import StorageAdapter


class S3Adapter(StorageAdapter):
    """S3 adapter for storing request/response bodies."""

    def __init__(
        self,
        session: aioboto3.Session,
        bucket: str,
        prefix: str = "llm-logs",
    ) -> None:
        self._session = session
        self._bucket = bucket
        self._prefix = prefix
        self._client = None

    @classmethod
    async def create(
        cls,
        bucket: str,
        prefix: str = "llm-logs",
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> "S3Adapter":
        """Create a new S3Adapter."""
        session = aioboto3.Session(
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        return cls(session, bucket, prefix)

    async def upload(
        self,
        request_id: UUID,
        request_body: dict[str, Any],
        response_content: str | dict[str, Any] | None,
    ) -> tuple[str, str | None]:
        """Upload request and response bodies to S3."""
        request_key = f"{self._prefix}/{request_id}/request.json"
        response_key = f"{self._prefix}/{request_id}/response.json" if response_content else None

        async with self._session.client("s3") as s3:
            await s3.put_object(
                Bucket=self._bucket,
                Key=request_key,
                Body=json.dumps(request_body, default=str),
                ContentType="application/json",
            )

            if response_content is not None:
                body = (
                    json.dumps(response_content, default=str)
                    if isinstance(response_content, dict)
                    else response_content
                )
                await s3.put_object(
                    Bucket=self._bucket,
                    Key=response_key,
                    Body=body,
                    ContentType="application/json",
                )

        return request_key, response_key

    async def close(self) -> None:
        """Close the S3 client (no-op for aioboto3 context-managed clients)."""
        pass
