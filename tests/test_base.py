"""Tests for the base module."""

import time

import pytest
from pydantic import BaseModel

from majordomo_llm.base import (
    LLM,
    TOKENS_PER_MILLION,
    LLMJSONResponse,
    LLMResponse,
    LLMStreamResponse,
    LLMStructuredResponse,
    Usage,
    _StreamState,
    canonicalize_json_schema_output,
)
from majordomo_llm.exceptions import ResponseParsingError

COUNTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "population": {"type": "integer"},
    },
    "required": ["name", "population"],
}


class TestJSONSchemaOutputHelpers:
    """Tests for JSON-schema output parsing and canonicalization."""

    def test_serializes_canonical_json(self):
        """Should sort keys and remove extra whitespace."""
        content = '{"population": 125000000, "name": "Japan"}'

        canonical = canonicalize_json_schema_output(content, COUNTRY_SCHEMA)

        assert canonical == '{"name":"Japan","population":125000000}'

    def test_repairs_markdown_fenced_json(self):
        """Should strip markdown fences before parsing."""
        content = '```json\n{"name":"Japan","population":125000000}\n```'

        canonical = canonicalize_json_schema_output(content, COUNTRY_SCHEMA)

        assert canonical == '{"name":"Japan","population":125000000}'

    def test_repairs_first_balanced_object(self):
        """Should extract the first balanced JSON object from surrounding text."""
        content = 'Here is the answer: {"name":"Japan","population":125000000} thanks.'

        canonical = canonicalize_json_schema_output(content, COUNTRY_SCHEMA)

        assert canonical == '{"name":"Japan","population":125000000}'

    def test_validation_error_includes_raw_content(self):
        """Should include raw content when schema validation fails."""
        raw_content = '{"name":"Japan","population":"many"}'

        with pytest.raises(ResponseParsingError) as exc_info:
            canonicalize_json_schema_output(raw_content, COUNTRY_SCHEMA)

        assert exc_info.value.raw_content == raw_content


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

        async def _get_response_impl(
            self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
            extra_headers=None,
        ):
            raise NotImplementedError()

        async def _get_response_stream_impl(
            self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
            extra_headers=None,
        ):
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

        async def _get_response_impl(
            self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
            extra_headers=None,
        ):
            raise NotImplementedError()

        async def _get_response_stream_impl(
            self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
            extra_headers=None,
        ):
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

        async def _get_response_impl(
            self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
            extra_headers=None,
        ):
            raise NotImplementedError()

        async def _get_response_stream_impl(
            self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
            extra_headers=None,
        ):
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
        state = _StreamState(
            input_tokens=10,
            output_tokens=5,
            cached_tokens=0,
            start_time=time.time(),
        )
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


class _RecordingLLM(LLM):
    """Test double that returns a canned LLMResponse and records prompts."""

    def __init__(self, content: str = "response", **kwargs):
        super().__init__(
            provider="test", model="test-model", input_cost=1.0, output_cost=2.0,
            **kwargs,
        )
        self.canned_content = content
        self.calls: list[str] = []
        self.schema_calls: list[str] = []

    async def _get_response_impl(
        self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
        extra_headers=None,
    ):
        self.calls.append(user_prompt)
        return LLMResponse(
            content=self.canned_content,
            input_tokens=10, output_tokens=20, cached_tokens=0,
            input_cost=0.01, output_cost=0.02, total_cost=0.03,
            response_time=0.1,
        )

    async def _get_response_stream_impl(self, *args, **kwargs):
        raise NotImplementedError()

    async def _get_json_schema_response(
        self, user_prompt, response_schema, system_prompt=None,
        schema_name="Response", schema_description=None,
        temperature=0.3, top_p=1.0, extra_headers=None,
    ):
        self.schema_calls.append(user_prompt)
        return LLMResponse(
            content=self.canned_content,
            input_tokens=10, output_tokens=20, cached_tokens=0,
            input_cost=0.01, output_cost=0.02, total_cost=0.03,
            response_time=0.1,
        )


class TestLLMWithoutHooks:
    """Regression guard: an LLM with no hook_pipeline behaves identically to today."""

    @pytest.mark.asyncio
    async def test_get_response_passes_through(self):
        llm = _RecordingLLM(content="hello")
        response = await llm.get_response("prompt")
        assert response.content == "hello"
        assert response.input_tokens == 10
        assert llm.calls == ["prompt"]

    @pytest.mark.asyncio
    async def test_get_json_schema_response_passes_through(self):
        llm = _RecordingLLM(content='{"name":"x","population":1}')
        response = await llm.get_json_schema_response(
            user_prompt="prompt", response_schema=COUNTRY_SCHEMA,
        )
        assert response.content == '{"name":"x","population":1}'
        assert llm.schema_calls == ["prompt"]


class TestLLMWithHooks:
    """Hooks attached to the LLM base class wrap text-producing calls."""

    @pytest.mark.asyncio
    async def test_redact_in_after_replaces_content_preserving_usage(self):
        from majordomo_llm import HookOutcome, HookPipeline

        class Hook:
            name = "redactor"

            async def before_call(self, prompt, ctx):
                return HookOutcome.pass_through(self.name)

            async def after_call(self, prompt, response, ctx):
                return HookOutcome.redact(self.name, "REDACTED", "test")

        pipeline = HookPipeline([Hook()])
        llm = _RecordingLLM(content="secret stuff", hook_pipeline=pipeline)
        response = await llm.get_response("prompt")
        assert response.content == "REDACTED"
        # usage preserved
        assert response.input_tokens == 10
        assert response.output_tokens == 20
        assert response.total_cost == 0.03

    @pytest.mark.asyncio
    async def test_block_in_before_prevents_impl_call(self):
        from majordomo_llm import HookBlocked, HookOutcome, HookPipeline

        class Hook:
            name = "blocker"

            async def before_call(self, prompt, ctx):
                return HookOutcome.block(self.name, "no")

            async def after_call(self, prompt, response, ctx):
                return HookOutcome.pass_through(self.name)

        pipeline = HookPipeline([Hook()])
        llm = _RecordingLLM(hook_pipeline=pipeline)
        with pytest.raises(HookBlocked):
            await llm.get_response("prompt")
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_caller_metadata_propagates_to_hook(self):
        from majordomo_llm import HookContext, HookOutcome, HookPipeline

        seen: list[HookContext] = []

        class Hook:
            name = "spy"

            async def before_call(self, prompt, ctx):
                seen.append(ctx)
                return HookOutcome.pass_through(self.name)

            async def after_call(self, prompt, response, ctx):
                return HookOutcome.pass_through(self.name)

        pipeline = HookPipeline([Hook()])
        llm = _RecordingLLM(hook_pipeline=pipeline)
        await llm.get_response("prompt", caller_metadata={"feature": "drafting"})
        assert seen[0].caller_metadata == {"feature": "drafting"}

    @pytest.mark.asyncio
    async def test_get_json_response_runs_hooks_through_get_response(self):
        from majordomo_llm import HookOutcome, HookPipeline

        class Hook:
            name = "rewriter"

            async def before_call(self, prompt, ctx):
                return HookOutcome.pass_through(self.name)

            async def after_call(self, prompt, response, ctx):
                return HookOutcome.redact(self.name, '{"final": true}', "rewrite")

        pipeline = HookPipeline([Hook()])
        llm = _RecordingLLM(content='{"initial": true}', hook_pipeline=pipeline)
        response = await llm.get_json_response("prompt")
        assert response.content == {"final": True}

    @pytest.mark.asyncio
    async def test_get_json_schema_response_runs_hooks(self):
        from majordomo_llm import HookOutcome, HookPipeline

        class Hook:
            name = "rewriter"

            async def before_call(self, prompt, ctx):
                return HookOutcome.pass_through(self.name)

            async def after_call(self, prompt, response, ctx):
                return HookOutcome.redact(
                    self.name, '{"name":"y","population":2}', "rewrite"
                )

        pipeline = HookPipeline([Hook()])
        llm = _RecordingLLM(
            content='{"name":"x","population":1}', hook_pipeline=pipeline,
        )
        response = await llm.get_json_schema_response(
            user_prompt="prompt", response_schema=COUNTRY_SCHEMA,
        )
        assert response.content == '{"name":"y","population":2}'
