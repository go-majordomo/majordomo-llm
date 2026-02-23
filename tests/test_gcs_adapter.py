"""Tests for GCS storage adapter."""

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from majordomo_llm.logging.adapters.gcs import GCSAdapter


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.upload = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def adapter(mock_client: AsyncMock) -> GCSAdapter:
    return GCSAdapter(client=mock_client, bucket="test-bucket", prefix="test-logs")


async def test_create_with_defaults() -> None:
    with patch("majordomo_llm.logging.adapters.gcs.Storage") as mock_storage:
        mock_storage.return_value = AsyncMock()
        adapter = await GCSAdapter.create(bucket="my-bucket")

        mock_storage.assert_called_once_with(service_file=None)
        assert adapter._bucket == "my-bucket"
        assert adapter._prefix == "llm-logs"


async def test_create_with_explicit_params() -> None:
    with patch("majordomo_llm.logging.adapters.gcs.Storage") as mock_storage:
        mock_storage.return_value = AsyncMock()
        adapter = await GCSAdapter.create(
            bucket="my-bucket",
            prefix="custom-prefix",
            project="my-project",
            service_account_path="/path/to/sa.json",
        )

        mock_storage.assert_called_once_with(service_file="/path/to/sa.json")
        assert adapter._bucket == "my-bucket"
        assert adapter._prefix == "custom-prefix"


async def test_upload(adapter: GCSAdapter, mock_client: AsyncMock) -> None:
    request_id = uuid4()
    request_body = {"model": "test-model", "prompt": "hello"}
    response_content = {"text": "world"}

    request_key, response_key = await adapter.upload(request_id, request_body, response_content)

    assert request_key == f"test-logs/{request_id}/request.json"
    assert response_key == f"test-logs/{request_id}/response.json"

    assert mock_client.upload.call_count == 2

    # Check request upload
    req_call = mock_client.upload.call_args_list[0]
    assert req_call.args[0] == "test-bucket"
    assert req_call.args[1] == f"test-logs/{request_id}/request.json"
    assert json.loads(req_call.args[2]) == request_body

    # Check response upload
    resp_call = mock_client.upload.call_args_list[1]
    assert resp_call.args[0] == "test-bucket"
    assert resp_call.args[1] == f"test-logs/{request_id}/response.json"
    assert json.loads(resp_call.args[2]) == response_content


async def test_upload_with_string_response(adapter: GCSAdapter, mock_client: AsyncMock) -> None:
    request_id = uuid4()
    request_body = {"model": "test-model"}
    response_content = "plain text response"

    request_key, response_key = await adapter.upload(request_id, request_body, response_content)

    assert response_key == f"test-logs/{request_id}/response.json"
    resp_call = mock_client.upload.call_args_list[1]
    assert resp_call.args[2] == "plain text response"


async def test_upload_with_none_response(adapter: GCSAdapter, mock_client: AsyncMock) -> None:
    request_id = uuid4()
    request_body = {"model": "test-model"}

    request_key, response_key = await adapter.upload(request_id, request_body, None)

    assert request_key == f"test-logs/{request_id}/request.json"
    assert response_key is None
    assert mock_client.upload.call_count == 1


async def test_close(adapter: GCSAdapter, mock_client: AsyncMock) -> None:
    await adapter.close()
    mock_client.close.assert_called_once()
