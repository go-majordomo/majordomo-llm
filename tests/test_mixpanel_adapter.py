"""Unit tests for MixpanelAdapter."""

from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from majordomo_llm.logging.adapters.mixpanel import MixpanelAdapter
from majordomo_llm.logging.models import LogEntry


def make_entry(**kwargs: object) -> LogEntry:
    defaults: dict[str, object] = {
        "request_id": UUID("12345678-1234-5678-1234-567812345678"),
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "timestamp": datetime(2024, 1, 1, 12, 0, 0),
        "response_time": 1.5,
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_tokens": 10,
        "input_cost": 0.001,
        "output_cost": 0.002,
        "total_cost": 0.003,
        "s3_request_key": "requests/abc.json",
        "s3_response_key": "responses/abc.json",
        "status": "success",
        "error_message": None,
        "api_key_hash": "abc123",
        "api_key_alias": "prod-key",
    }
    defaults.update(kwargs)
    return LogEntry(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_create_default() -> None:
    with patch("majordomo_llm.logging.adapters.mixpanel.Mixpanel") as MockMixpanel:
        adapter = await MixpanelAdapter.create("test-token")
        MockMixpanel.assert_called_once_with("test-token")
        assert isinstance(adapter, MixpanelAdapter)


@pytest.mark.asyncio
async def test_create_with_api_host() -> None:
    with (
        patch("majordomo_llm.logging.adapters.mixpanel.Mixpanel") as MockMixpanel,
        patch("majordomo_llm.logging.adapters.mixpanel.Consumer") as MockConsumer,
    ):
        consumer_instance = MagicMock()
        MockConsumer.return_value = consumer_instance
        adapter = await MixpanelAdapter.create("test-token", api_host="https://api-eu.mixpanel.com")
        MockConsumer.assert_called_once_with(api_host="https://api-eu.mixpanel.com")
        MockMixpanel.assert_called_once_with("test-token", consumer=consumer_instance)
        assert isinstance(adapter, MixpanelAdapter)


@pytest.mark.asyncio
async def test_insert_sends_correct_event() -> None:
    mock_mp = MagicMock()
    adapter = MixpanelAdapter(mock_mp)
    entry = make_entry()

    await adapter.insert(entry)

    mock_mp.track.assert_called_once()
    call_args = mock_mp.track.call_args
    distinct_id, event_name, properties = call_args[0]

    assert distinct_id == "prod-key"
    assert event_name == "llm_request"
    assert properties["request_id"] == "12345678-1234-5678-1234-567812345678"
    assert properties["provider"] == "anthropic"
    assert properties["model"] == "claude-sonnet-4-20250514"
    assert properties["status"] == "success"
    assert properties["timestamp"] == "2024-01-01T12:00:00"
    assert properties["response_time"] == 1.5
    assert properties["input_tokens"] == 100
    assert properties["output_tokens"] == 50
    assert properties["cached_tokens"] == 10
    assert properties["input_cost"] == 0.001
    assert properties["output_cost"] == 0.002
    assert properties["total_cost"] == 0.003
    assert properties["s3_request_key"] == "requests/abc.json"
    assert properties["s3_response_key"] == "responses/abc.json"
    assert properties["api_key_hash"] == "abc123"
    assert properties["api_key_alias"] == "prod-key"


@pytest.mark.asyncio
async def test_insert_excludes_none_properties() -> None:
    mock_mp = MagicMock()
    adapter = MixpanelAdapter(mock_mp)
    entry = make_entry(
        error_message=None,
        s3_request_key=None,
        s3_response_key=None,
        api_key_hash=None,
        api_key_alias=None,
        response_time=None,
        input_tokens=None,
        output_tokens=None,
        cached_tokens=None,
        input_cost=None,
        output_cost=None,
        total_cost=None,
    )

    await adapter.insert(entry)

    _, _, properties = mock_mp.track.call_args[0]
    assert "error_message" not in properties
    assert "s3_request_key" not in properties
    assert "s3_response_key" not in properties
    assert "api_key_hash" not in properties
    assert "api_key_alias" not in properties
    assert "response_time" not in properties
    assert "input_tokens" not in properties


@pytest.mark.asyncio
async def test_insert_falls_back_to_unknown_distinct_id() -> None:
    mock_mp = MagicMock()
    adapter = MixpanelAdapter(mock_mp)
    entry = make_entry(api_key_alias=None)

    await adapter.insert(entry)

    distinct_id, _, _ = mock_mp.track.call_args[0]
    assert distinct_id == "unknown"


@pytest.mark.asyncio
async def test_close_is_noop() -> None:
    mock_mp = MagicMock()
    adapter = MixpanelAdapter(mock_mp)
    await adapter.close()  # Should not raise
    mock_mp.assert_not_called()
