"""Tests for the base module."""

import time

import pytest
from pydantic import BaseModel

from majordomo_llm.base import (
    LLM,
    LLMResponse,
    LLMJSONResponse,
    LLMStreamResponse,
    LLMStructuredResponse,
    Usage,
    TOKENS_PER_MILLION,
    _StreamState,
)


class TestUsage:
    """Tests for Usage dataclass."""

    def test_usage_stores_all_fields(self):
        """Should store all usage metrics."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=10,
            input_cost=0.0003,
            output_cost=0.00075,
            total_cost=0.00105,
            response_time=1.5,
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 10
        assert usage.input_cost == 0.0003
        assert usage.output_cost == 0.00075
        assert usage.total_cost == 0.00105
        assert usage.response_time == 1.5


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_includes_content_and_usage(self):
        """Should include content and inherit usage fields."""
        response = LLMResponse(
            content="Hello, world!",
            input_tokens=10,
            output_tokens=5,
            cached_tokens=0,
            input_cost=0.00003,
            output_cost=0.000075,
            total_cost=0.000105,
            response_time=0.5,
        )

        assert response.content == "Hello, world!"
        assert response.input_tokens == 10
        assert response.output_tokens == 5


class TestLLMJSONResponse:
    """Tests for LLMJSONResponse dataclass."""

    def test_content_is_dict(self):
        """Content should be a dictionary."""
        response = LLMJSONResponse(
            content={"key": "value", "number": 42},
            input_tokens=20,
            output_tokens=10,
            cached_tokens=0,
            input_cost=0.00006,
            output_cost=0.00015,
            total_cost=0.00021,
            response_time=0.8,
        )

        assert response.content == {"key": "value", "number": 42}
        assert response.content["key"] == "value"


class TestLLMStructuredResponse:
    """Tests for LLMStructuredResponse dataclass."""

    def test_content_is_pydantic_model(self):
        """Content should be a Pydantic model instance."""

        class Person(BaseModel):
            name: str
            age: int

        person = Person(name="Alice", age=30)
        response = LLMStructuredResponse(
            content=person,
            input_tokens=30,
            output_tokens=15,
            cached_tokens=5,
            input_cost=0.00009,
            output_cost=0.000225,
            total_cost=0.000315,
            response_time=1.0,
        )

        assert response.content.name == "Alice"
        assert response.content.age == 30


class TestLLMCostCalculation:
    """Tests for LLM._calculate_costs method."""

    class ConcreteLLM(LLM):
        """Concrete implementation for testing abstract base class."""

        async def get_response(self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0):
            raise NotImplementedError()

        async def get_response_stream(self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0):
            raise NotImplementedError()

    def test_calculates_costs_correctly(self):
        """Should calculate costs based on tokens and rates."""
        llm = self.ConcreteLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,  # $3 per million tokens
            output_cost=15.0,  # $15 per million tokens
        )

        input_cost, output_cost, total_cost = llm._calculate_costs(
            input_tokens=1000,
            output_tokens=500,
        )

        expected_input = 1000 * 3.0 / TOKENS_PER_MILLION
        expected_output = 500 * 15.0 / TOKENS_PER_MILLION

        assert input_cost == expected_input
        assert output_cost == expected_output
        assert total_cost == expected_input + expected_output

    def test_handles_zero_tokens(self):
        """Should handle zero tokens gracefully."""
        llm = self.ConcreteLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,
            output_cost=15.0,
        )

        input_cost, output_cost, total_cost = llm._calculate_costs(
            input_tokens=0,
            output_tokens=0,
        )

        assert input_cost == 0.0
        assert output_cost == 0.0
        assert total_cost == 0.0

    def test_handles_large_token_counts(self):
        """Should handle large token counts correctly."""
        llm = self.ConcreteLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,
            output_cost=15.0,
        )

        input_cost, output_cost, total_cost = llm._calculate_costs(
            input_tokens=1_000_000,  # 1 million tokens
            output_tokens=1_000_000,
        )

        assert input_cost == 3.0  # Exactly $3
        assert output_cost == 15.0  # Exactly $15
        assert total_cost == 18.0


class TestLLMFullModelName:
    """Tests for LLM.get_full_model_name method."""

    class ConcreteLLM(LLM):
        """Concrete implementation for testing."""

        async def get_response(self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0):
            raise NotImplementedError()

        async def get_response_stream(self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0):
            raise NotImplementedError()

    def test_returns_provider_colon_model(self):
        """Should return 'provider:model' format."""
        llm = self.ConcreteLLM(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            input_cost=3.0,
            output_cost=15.0,
        )

        assert llm.get_full_model_name() == "anthropic:claude-sonnet-4-20250514"


class TestLLMStreamResponse:
    """Tests for LLMStreamResponse async streaming wrapper."""

    class ConcreteLLM(LLM):
        """Concrete implementation for testing abstract base class."""

        async def get_response(self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0):
            raise NotImplementedError()

        async def get_response_stream(self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0):
            raise NotImplementedError()

    @staticmethod
    async def _mock_stream():
        yield "Hello"
        yield " "
        yield "world"

    def _make_llm(self):
        return self.ConcreteLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,
            output_cost=15.0,
        )

    def _make_stream_response(self, stream=None):
        llm = self._make_llm()
        state = _StreamState(input_tokens=10, output_tokens=5, cached_tokens=0, start_time=time.time())
        if stream is None:
            stream = self._mock_stream()
        return LLMStreamResponse(stream=stream, state=state, llm=llm)

    @pytest.mark.asyncio
    async def test_iterating_yields_chunks(self):
        """Iterating over the stream should yield each chunk in order."""
        stream = self._make_stream_response()
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert chunks == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_usage_populated_after_iteration(self):
        """Usage should be populated with correct values after consuming the stream."""
        stream = self._make_stream_response()
        async for _ in stream:
            pass

        assert stream.usage is not None
        assert stream.usage.input_tokens == 10
        assert stream.usage.output_tokens == 5
        assert stream.usage.cached_tokens == 0
        assert stream.usage.input_cost == 10 * 3.0 / TOKENS_PER_MILLION
        assert stream.usage.output_cost == 5 * 15.0 / TOKENS_PER_MILLION
        assert stream.usage.total_cost == stream.usage.input_cost + stream.usage.output_cost

    @pytest.mark.asyncio
    async def test_usage_is_none_before_consumption(self):
        """Usage should be None before the stream is consumed."""
        stream = self._make_stream_response()

        assert stream.usage is None

    @pytest.mark.asyncio
    async def test_collect_returns_llm_response(self):
        """collect() should return an LLMResponse with correct content and usage."""
        stream = self._make_stream_response()
        response = await stream.collect()

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello world"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.cached_tokens == 0
        assert response.input_cost == 10 * 3.0 / TOKENS_PER_MILLION
        assert response.output_cost == 5 * 15.0 / TOKENS_PER_MILLION
        assert response.total_cost == response.input_cost + response.output_cost

    @pytest.mark.asyncio
    async def test_on_complete_callback_fires(self):
        """_on_complete callback should be called with Usage and content after iteration."""
        stream = self._make_stream_response()
        callback_args = {}

        def on_complete(usage, content):
            callback_args["usage"] = usage
            callback_args["content"] = content

        stream._on_complete = on_complete

        async for _ in stream:
            pass

        assert "usage" in callback_args
        assert "content" in callback_args
        assert isinstance(callback_args["usage"], Usage)
        assert callback_args["usage"].input_tokens == 10
        assert callback_args["usage"].output_tokens == 5
        assert callback_args["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_on_error_callback_fires(self):
        """_on_error callback should be called when the stream raises an exception."""

        async def failing_stream():
            yield "partial"
            raise RuntimeError("stream failed")

        stream = self._make_stream_response(stream=failing_stream())
        error_args = {}

        def on_error(exc):
            error_args["exception"] = exc

        stream._on_error = on_error

        with pytest.raises(RuntimeError, match="stream failed"):
            async for _ in stream:
                pass

        assert "exception" in error_args
        assert isinstance(error_args["exception"], RuntimeError)
        assert str(error_args["exception"]) == "stream failed"
