"""Mixpanel analytics adapter."""

import asyncio

from mixpanel import Consumer, Mixpanel

from majordomo_llm.logging.interfaces import DatabaseAdapter
from majordomo_llm.logging.models import LogEntry


class MixpanelAdapter(DatabaseAdapter):
    """
    Mixpanel adapter for logging LLM requests as analytics events.

    Each call to insert() sends an "llm_request" event to Mixpanel with
    all LogEntry fields as event properties.

    Example:
        >>> db = await MixpanelAdapter.create("your-mixpanel-token")
        >>> logged_llm = LoggingLLM(llm, db)

        >>> # For EU/India data residency:
        >>> db = await MixpanelAdapter.create("your-token", api_host="https://api-eu.mixpanel.com")
    """

    def __init__(self, mp: Mixpanel) -> None:
        self._mp = mp

    @classmethod
    async def create(cls, token: str, api_host: str | None = None) -> "MixpanelAdapter":
        """Create a new MixpanelAdapter.

        Args:
            token: Mixpanel project token.
            api_host: Optional custom API host for EU/India data residency.

        Returns:
            A configured MixpanelAdapter instance.
        """
        if api_host is not None:
            mp = Mixpanel(token, consumer=Consumer(api_host=api_host))
        else:
            mp = Mixpanel(token)
        return cls(mp)

    async def insert(self, entry: LogEntry) -> None:
        """Send a log entry to Mixpanel as an "llm_request" event."""
        distinct_id = entry.api_key_alias or "unknown"
        raw_properties: dict[str, object] = {
            "request_id": str(entry.request_id),
            "provider": entry.provider,
            "model": entry.model,
            "status": entry.status,
            "error_message": entry.error_message,
            "timestamp": entry.timestamp.isoformat(),
            "response_time": entry.response_time,
            "input_tokens": entry.input_tokens,
            "output_tokens": entry.output_tokens,
            "cached_tokens": entry.cached_tokens,
            "input_cost": entry.input_cost,
            "output_cost": entry.output_cost,
            "total_cost": entry.total_cost,
            "s3_request_key": entry.s3_request_key,
            "s3_response_key": entry.s3_response_key,
            "api_key_hash": entry.api_key_hash,
            "api_key_alias": entry.api_key_alias,
        }
        properties = {k: v for k, v in raw_properties.items() if v is not None}
        await asyncio.to_thread(self._mp.track, distinct_id, "llm_request", properties)

    async def close(self) -> None:
        """No-op: the Mixpanel SDK has no persistent connection to close."""
        pass
