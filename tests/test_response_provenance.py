"""Responses name the provider and model that served them.

Direct calls through each real provider SDK, and the reported cascade case, are
in ``test_sdk_payloads.py::TestResponseProvenance``. This module covers the
paths that rebuild or pass along a response: the cascade's failover
bookkeeping, hook rewrites, the JSON and structured wrappers, streaming, and
``LoggingLLM``.
"""

from typing import Any

import pytest
from pydantic import BaseModel
from tenacity import Future, RetryError

from majordomo_llm import LLMCascade
from majordomo_llm.base import LLM, LLMResponse, LLMStreamResponse, _StreamState
from majordomo_llm.exceptions import (
    EmptyStructuredResponseError,
    ProviderError,
    ResponseParsingError,
)
from majordomo_llm.hooks import HookContext, HookOutcome, HookPipeline
from majordomo_llm.logging.interfaces import DatabaseAdapter
from majordomo_llm.logging.models import LogEntry
from majordomo_llm.logging.wrapper import LoggingLLM


class Country(BaseModel):
    name: str


def _response(content: str = "answer", **kwargs: Any) -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=10,
        output_tokens=20,
        cached_tokens=0,
        input_cost=0.01,
        output_cost=0.02,
        total_cost=0.03,
        response_time=0.1,
        **kwargs,
    )


class _FakeLLM(LLM):
    """A provider that answers with a canned response or raises ``error``."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        content: str = "answer",
        error: Exception | None = None,
        hook_pipeline: HookPipeline | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            model=model,
            input_cost=1.0,
            output_cost=2.0,
            hook_pipeline=hook_pipeline,
        )
        self.content = content
        self.error = error

    async def _get_response_impl(self, *args: Any, **kwargs: Any) -> LLMResponse:
        if self.error is not None:
            raise self.error
        return _response(self.content, tool_use_cost=0.005)

    async def _get_response_stream_impl(self, *args: Any, **kwargs: Any) -> LLMStreamResponse:
        if self.error is not None:
            raise self.error

        async def chunks():
            yield self.content

        return LLMStreamResponse(chunks(), _StreamState(), self)

    async def _get_json_schema_response(self, *args: Any, **kwargs: Any) -> LLMResponse:
        if self.error is not None:
            raise self.error
        return _response(self.content)


def _cascade(*members: _FakeLLM) -> LLMCascade:
    """A real LLMCascade over the same (provider, model) pairs, served by ``members``.

    The cascade is built through the factory, so its own metadata is what a
    caller would get; its members are then replaced so no SDK is involved.
    """
    cascade = LLMCascade([(m.provider, m.model) for m in members], api_key="test-key")
    cascade.llms = list(members)
    return cascade


def _down(provider: str) -> ProviderError:
    return ProviderError(f"{provider} down", provider=provider)


class TestDirectCalls:
    async def test_text(self):
        response = await _FakeLLM("anthropic", "claude-sonnet-5-5").get_response("hi")

        assert (response.provider, response.model) == ("anthropic", "claude-sonnet-5-5")
        assert response.attempted == [("anthropic", "claude-sonnet-5-5")]

    async def test_json(self):
        llm = _FakeLLM("openai", "gpt-5.5", content='{"name": "France"}')

        response = await llm.get_json_response("hi")

        assert (response.provider, response.model) == ("openai", "gpt-5.5")
        assert response.attempted == [("openai", "gpt-5.5")]

    async def test_structured(self):
        llm = _FakeLLM("gemini", "gemini-3.8-flash", content='{"name": "France"}')

        response = await llm.get_structured_json_response(Country, "hi")

        assert response.content == Country(name="France")
        assert (response.provider, response.model) == ("gemini", "gemini-3.8-flash")

    async def test_stream_before_and_after_it_finishes(self):
        stream = await _FakeLLM("cohere", "command-a-03-2025").get_response_stream("hi")

        assert (stream.provider, stream.model) == ("cohere", "command-a-03-2025")
        assert stream.usage is None

        collected = await stream.collect()

        assert stream.usage is not None
        assert (stream.usage.provider, stream.usage.model) == ("cohere", "command-a-03-2025")
        assert (collected.provider, collected.model) == ("cohere", "command-a-03-2025")
        assert collected.attempted == [("cohere", "command-a-03-2025")]

    async def test_a_response_that_names_its_origin_is_kept(self):
        """The base class fills only empty fields, so a member's values survive a cascade."""

        class _SelfReporting(_FakeLLM):
            async def _get_response_impl(self, *args: Any, **kwargs: Any) -> LLMResponse:
                return _response(provider="openai", model="gpt-5.5", attempted=[("openai", "x")])

        response = await _SelfReporting("cascade", "primary").get_response("hi")

        assert (response.provider, response.model) == ("openai", "gpt-5.5")
        assert response.attempted == [("openai", "x")]


class TestCascade:
    async def test_primary_answers(self):
        cascade = _cascade(
            _FakeLLM("anthropic", "claude-sonnet-5-5", content="from anthropic"),
            _FakeLLM("openai", "gpt-5.5"),
        )

        response = await cascade.get_response("hi")

        assert response.content == "from anthropic"
        assert (response.provider, response.model) == ("anthropic", "claude-sonnet-5-5")
        assert response.attempted == [("anthropic", "claude-sonnet-5-5")]

    async def test_falls_through_to_last_member(self):
        cascade = _cascade(
            _FakeLLM("anthropic", "claude-sonnet-5-5", error=_down("anthropic")),
            _FakeLLM("openai", "gpt-5.5", error=_down("openai")),
            _FakeLLM("gemini", "gemini-3.8-flash", content="from gemini"),
        )

        response = await cascade.get_response("hi")

        assert response.content == "from gemini"
        assert (response.provider, response.model) == ("gemini", "gemini-3.8-flash")
        assert response.attempted == [
            ("anthropic", "claude-sonnet-5-5"),
            ("openai", "gpt-5.5"),
            ("gemini", "gemini-3.8-flash"),
        ]

    async def test_exhausted_retries_count_as_an_attempt(self):
        exhausted = RetryError(
            Future.construct(attempt_number=3, value=_down("anthropic"), has_exception=True)
        )
        cascade = _cascade(
            _FakeLLM("anthropic", "claude-sonnet-5-5", error=exhausted),
            _FakeLLM("openai", "gpt-5.5"),
        )

        response = await cascade.get_response("hi")

        assert response.attempted == [("anthropic", "claude-sonnet-5-5"), ("openai", "gpt-5.5")]

    async def test_structured_falls_through(self):
        cascade = _cascade(
            _FakeLLM("anthropic", "claude-sonnet-5-5", error=_down("anthropic")),
            _FakeLLM("openai", "gpt-5.5", content='{"name": "France"}'),
        )

        response = await cascade.get_structured_json_response(Country, "hi")

        assert (response.provider, response.model) == ("openai", "gpt-5.5")
        assert response.attempted == [("anthropic", "claude-sonnet-5-5"), ("openai", "gpt-5.5")]

    async def test_json_falls_through(self):
        cascade = _cascade(
            _FakeLLM("anthropic", "claude-sonnet-5-5", error=_down("anthropic")),
            _FakeLLM("openai", "gpt-5.5", content='{"name": "France"}'),
        )

        response = await cascade.get_json_response("hi")

        assert (response.provider, response.model) == ("openai", "gpt-5.5")

    async def test_stream_falls_through(self):
        cascade = _cascade(
            _FakeLLM("anthropic", "claude-sonnet-5-5", error=_down("anthropic")),
            _FakeLLM("openai", "gpt-5.5", content="streamed"),
        )

        stream = await cascade.get_response_stream("hi")
        collected = await stream.collect()

        assert collected.content == "streamed"
        assert (stream.provider, stream.model) == ("openai", "gpt-5.5")
        assert stream.usage is not None
        assert (stream.usage.provider, stream.usage.model) == ("openai", "gpt-5.5")
        assert stream.usage.attempted == [
            ("anthropic", "claude-sonnet-5-5"),
            ("openai", "gpt-5.5"),
        ]
        assert collected.attempted == stream.usage.attempted


class TestBilledErrors:
    """A parse error that ends a cascade names the members tried, like a response."""

    def _empty(self) -> EmptyStructuredResponseError:
        return EmptyStructuredResponseError("empty", raw_content="{}", usage=_response("{}"))

    async def test_json_parse_error_after_failover(self):
        """get_json_response parses the member's answer, so its usage names that member."""
        cascade = _cascade(
            _FakeLLM("anthropic", "claude-sonnet-5-5", error=_down("anthropic")),
            _FakeLLM("openai", "gpt-5.5", content="not json"),
        )

        with pytest.raises(ResponseParsingError) as caught:
            await cascade.get_json_response("hi")

        usage = caught.value.usage
        assert usage is not None
        assert (usage.provider, usage.model) == ("openai", "gpt-5.5")
        assert usage.attempted == [("anthropic", "claude-sonnet-5-5"), ("openai", "gpt-5.5")]
        assert usage.total_cost == 0.03

    async def test_parse_error_inside_exhausted_retries(self):
        empty = self._empty()
        exhausted = RetryError(Future.construct(attempt_number=3, value=empty, has_exception=True))
        cascade = _cascade(
            _FakeLLM("anthropic", "claude-sonnet-5-5", error=_down("anthropic")),
            _FakeLLM("openai", "gpt-5.5", error=exhausted),
        )

        with pytest.raises(RetryError):
            await cascade.get_json_schema_response("hi", {"type": "object"})

        assert empty.usage is not None
        assert (empty.usage.provider, empty.usage.model) == ("openai", "gpt-5.5")
        assert empty.usage.attempted == [
            ("anthropic", "claude-sonnet-5-5"),
            ("openai", "gpt-5.5"),
        ]


class _UppercaseHook:
    """Rewrites every response, which makes the base class rebuild it."""

    name = "uppercase"

    async def before_call(self, prompt: str, ctx: HookContext) -> HookOutcome:
        return HookOutcome.pass_through(self.name)

    async def after_call(self, prompt: str, response: str, ctx: HookContext) -> HookOutcome:
        return HookOutcome.redact(self.name, response.upper(), "uppercase")


class TestHookRewrite:
    async def test_rewritten_response_keeps_every_other_field(self):
        llm = _FakeLLM(
            "anthropic",
            "claude-sonnet-5-5",
            content="quiet",
            hook_pipeline=HookPipeline([_UppercaseHook()]),
        )

        response = await llm.get_response("hi")

        assert response.content == "QUIET"
        assert (response.provider, response.model) == ("anthropic", "claude-sonnet-5-5")
        assert response.attempted == [("anthropic", "claude-sonnet-5-5")]
        # Dropped by the rebuild before this change.
        assert response.tool_use_cost == 0.005


class _RecordingDatabase(DatabaseAdapter):
    def __init__(self) -> None:
        self.entries: list[LogEntry] = []

    async def insert(self, entry: LogEntry) -> None:
        self.entries.append(entry)

    async def close(self) -> None:
        pass


class TestLoggingLLM:
    @pytest.fixture
    def database(self) -> _RecordingDatabase:
        return _RecordingDatabase()

    def _fallback_cascade(self) -> LLMCascade:
        return _cascade(
            _FakeLLM("anthropic", "claude-sonnet-5-5", error=_down("anthropic")),
            _FakeLLM("openai", "gpt-5.5"),
        )

    async def test_logs_the_member_that_answered(self, database):
        logged = LoggingLLM(self._fallback_cascade(), database)

        await logged.get_response("hi")
        await logged.flush()

        [entry] = database.entries
        assert (entry.provider, entry.model) == ("openai", "gpt-5.5")

    async def test_logs_the_member_that_streamed(self, database):
        logged = LoggingLLM(self._fallback_cascade(), database)

        stream = await logged.get_response_stream("hi")
        await stream.collect()
        await logged.flush()

        [entry] = database.entries
        assert (entry.provider, entry.model) == ("openai", "gpt-5.5")

    async def test_failed_call_is_logged_against_the_configured_llm(self, database):
        cascade = _cascade(
            _FakeLLM("anthropic", "claude-sonnet-5-5", error=_down("anthropic")),
        )
        logged = LoggingLLM(cascade, database)

        with pytest.raises(ProviderError):
            await logged.get_response("hi")
        await logged.flush()

        [entry] = database.entries
        assert entry.status == "error"
        assert (entry.provider, entry.model) == ("cascade", "claude-sonnet-5-5")
