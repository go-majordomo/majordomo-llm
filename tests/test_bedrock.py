"""Tests for the Amazon Bedrock provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from majordomo_llm.base import TOKENS_PER_MILLION
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers import Bedrock


class CountryInfo(BaseModel):
    """Test model for structured responses."""

    name: str
    capital: str
    population: int


COUNTRY_SCHEMA = CountryInfo.model_json_schema()


def _make_bedrock(model: str = "us.anthropic.claude-sonnet-4-5-v1:0") -> Bedrock:
    return Bedrock(
        model=model,
        input_cost=3.0,
        output_cost=15.0,
        api_key="test-key",
        region="us-east-1",
    )


def _install_mock_client(bedrock: Bedrock) -> MagicMock:
    """Replace bedrock._client with an async context manager returning a mock."""
    client = MagicMock()
    client.converse = AsyncMock()
    client.converse_stream = AsyncMock()

    @asynccontextmanager
    async def fake_client():
        yield client

    bedrock._client = fake_client  # type: ignore[method-assign]
    return client


def _converse_response(text: str = "Paris is the capital of France.") -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "usage": {"inputTokens": 25, "outputTokens": 10, "cacheReadInputTokens": 0},
        "stopReason": "end_turn",
    }


def _tool_use_response(name: str, value: dict) -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "t1", "name": name, "input": value}}],
            }
        },
        "usage": {"inputTokens": 50, "outputTokens": 30, "cacheReadInputTokens": 5},
        "stopReason": "tool_use",
    }


class TestBedrockInit:
    def test_raises_configuration_error_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                Bedrock(
                    model="us.anthropic.claude-sonnet-4-5-v1:0",
                    input_cost=3.0,
                    output_cost=15.0,
                    region="us-east-1",
                )
            assert "AWS_BEARER_TOKEN_BEDROCK" in str(exc_info.value)

    def test_raises_configuration_error_without_region(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                Bedrock(
                    model="us.anthropic.claude-sonnet-4-5-v1:0",
                    input_cost=3.0,
                    output_cost=15.0,
                    api_key="test-key",
                )
            assert "region" in str(exc_info.value).lower()

    def test_region_from_env_var(self):
        with patch.dict("os.environ", {"AWS_REGION": "us-west-2"}, clear=True):
            llm = Bedrock(
                model="us.anthropic.claude-sonnet-4-5-v1:0",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )
            assert llm.region == "us-west-2"

    def test_sets_provider_name(self):
        llm = _make_bedrock()
        assert llm.provider == "bedrock"

    def test_stores_model_and_costs(self):
        llm = _make_bedrock()
        assert llm.model == "us.anthropic.claude-sonnet-4-5-v1:0"
        assert llm.input_cost == 3.0
        assert llm.output_cost == 15.0


class TestBedrockGetResponse:
    async def test_returns_text_content(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        response = await llm.get_response("What is the capital of France?")

        assert response.content == "Paris is the capital of France."

    async def test_returns_correct_token_counts(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        response = await llm.get_response("Test prompt")

        assert response.input_tokens == 25
        assert response.output_tokens == 10
        assert response.cached_tokens == 0

    async def test_calculates_costs_correctly(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        response = await llm.get_response("Test prompt")

        expected_input_cost = 25 * 3.0 / TOKENS_PER_MILLION
        expected_output_cost = 10 * 15.0 / TOKENS_PER_MILLION
        assert response.input_cost == expected_input_cost
        assert response.output_cost == expected_output_cost
        assert response.total_cost == expected_input_cost + expected_output_cost

    async def test_passes_temperature_and_top_p(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        await llm.get_response("Test prompt", temperature=0.7, top_p=0.9)

        kwargs = client.converse.call_args.kwargs
        assert kwargs["inferenceConfig"]["temperature"] == 0.7
        assert kwargs["inferenceConfig"]["topP"] == 0.9

    async def test_omits_temperature_when_unsupported(self):
        llm = Bedrock(
            model="us.anthropic.claude-sonnet-4-5-v1:0",
            input_cost=3.0,
            output_cost=15.0,
            supports_temperature_top_p=False,
            api_key="test-key",
            region="us-east-1",
        )
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        await llm.get_response("Test prompt", temperature=0.7, top_p=0.9)

        kwargs = client.converse.call_args.kwargs
        assert "temperature" not in kwargs["inferenceConfig"]
        assert "topP" not in kwargs["inferenceConfig"]

    async def test_uses_default_system_prompt(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        await llm.get_response("Test prompt")

        kwargs = client.converse.call_args.kwargs
        assert "helpful assistant" in kwargs["system"][0]["text"]


# Substring not in _BEDROCK_STRUCTURED_OUTPUTS_SUPPORTED — exercises the
# tool-calling fallback path in the tests below.
_TOOL_CALLING_FALLBACK_MODEL = "moonshotai.kimi-k2.5"


class TestBedrockStructuredResponse:
    async def test_extracts_tool_use_content(self):
        llm = _make_bedrock(model=_TOOL_CALLING_FALLBACK_MODEL)
        client = _install_mock_client(llm)
        client.converse.return_value = _tool_use_response(
            "CountryInfo",
            {"name": "France", "capital": "Paris", "population": 67000000},
        )

        response = await llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        assert response.content.name == "France"
        assert response.content.capital == "Paris"
        assert response.content.population == 67000000

    async def test_forces_tool_choice(self):
        llm = _make_bedrock(model=_TOOL_CALLING_FALLBACK_MODEL)
        client = _install_mock_client(llm)
        client.converse.return_value = _tool_use_response(
            "CountryInfo",
            {"name": "France", "capital": "Paris", "population": 67000000},
        )

        await llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        kwargs = client.converse.call_args.kwargs
        assert kwargs["toolConfig"]["toolChoice"] == {"tool": {"name": "CountryInfo"}}
        assert kwargs["toolConfig"]["tools"][0]["toolSpec"]["name"] == "CountryInfo"

    async def test_omits_tool_choice_for_llama4(self):
        """Llama 4 on Bedrock rejects toolChoice.tool; we expose the tool but
        do not force it (model is steered via the system prompt instead)."""
        llm = _make_bedrock(model="us.meta.llama4-scout-17b-instruct-v1:0")
        client = _install_mock_client(llm)
        client.converse.return_value = _tool_use_response(
            "CountryInfo",
            {"name": "France", "capital": "Paris", "population": 67000000},
        )

        await llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        kwargs = client.converse.call_args.kwargs
        assert "toolChoice" not in kwargs["toolConfig"]
        # The tool itself must still be exposed so the model can call it.
        assert kwargs["toolConfig"]["tools"][0]["toolSpec"]["name"] == "CountryInfo"

    async def test_json_schema_response_uses_schema_tool(self):
        llm = _make_bedrock(model=_TOOL_CALLING_FALLBACK_MODEL)
        client = _install_mock_client(llm)
        client.converse.return_value = _tool_use_response(
            "CountryInfo",
            {"name": "France", "capital": "Paris", "population": 67000000},
        )

        await llm.get_json_schema_response(
            user_prompt="Tell me about France",
            response_schema=COUNTRY_SCHEMA,
            schema_name="CountryInfo",
        )

        kwargs = client.converse.call_args.kwargs
        spec = kwargs["toolConfig"]["tools"][0]["toolSpec"]
        assert spec["name"] == "CountryInfo"


class TestBedrockNativeStructuredOutputs:
    """Tests for Bedrock's native Structured Outputs feature (outputConfig)."""

    async def test_uses_output_config_for_supported_model(self):
        """Claude models should use outputConfig, not toolConfig."""
        llm = _make_bedrock()  # default us.anthropic.claude-* — supported
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response(
            text='{"name": "France", "capital": "Paris", "population": 67000000}'
        )

        await llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        kwargs = client.converse.call_args.kwargs
        assert "toolConfig" not in kwargs
        assert "outputConfig" in kwargs
        text_format = kwargs["outputConfig"]["textFormat"]
        assert text_format["type"] == "json_schema"
        assert text_format["structure"]["jsonSchema"]["name"] == "CountryInfo"

    async def test_serializes_schema_as_json_string(self):
        """The schema field must be a JSON string, not a dict — Bedrock requires it.

        Bedrock also requires ``additionalProperties: false`` and every property
        listed in ``required`` on each object node (same as OpenAI strict mode),
        so the schema that lands in the request is the strict-normalized form.
        """
        import json

        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response(
            text='{"name": "France", "capital": "Paris", "population": 67000000}'
        )

        await llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        schema_field = client.converse.call_args.kwargs["outputConfig"]["textFormat"][
            "structure"
        ]["jsonSchema"]["schema"]
        assert isinstance(schema_field, str)
        parsed = json.loads(schema_field)
        assert parsed["additionalProperties"] is False
        assert set(parsed["required"]) == {"name", "capital", "population"}

    async def test_parses_text_content_as_json(self):
        """Native path returns JSON in message text, not a toolUse block."""
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response(
            text='{"name": "France", "capital": "Paris", "population": 67000000}'
        )

        response = await llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        assert response.content.name == "France"
        assert response.content.capital == "Paris"
        assert response.content.population == 67000000

    async def test_get_json_schema_response_uses_native_path(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response(
            text='{"name": "France", "capital": "Paris", "population": 67000000}'
        )

        await llm.get_json_schema_response(
            user_prompt="Tell me about France",
            response_schema=COUNTRY_SCHEMA,
            schema_name="CountryInfo",
        )

        kwargs = client.converse.call_args.kwargs
        assert "toolConfig" not in kwargs
        assert kwargs["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["name"] == (
            "CountryInfo"
        )

    async def test_falls_back_to_tool_calling_for_unsupported_model(self):
        """Models not in the allowlist should still get the tool-calling path."""
        llm = _make_bedrock(model=_TOOL_CALLING_FALLBACK_MODEL)
        client = _install_mock_client(llm)
        client.converse.return_value = _tool_use_response(
            "CountryInfo",
            {"name": "France", "capital": "Paris", "population": 67000000},
        )

        await llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        kwargs = client.converse.call_args.kwargs
        assert "outputConfig" not in kwargs
        assert "toolConfig" in kwargs


class TestBedrockStream:
    async def _stream_response(self):
        async def stream_iter():
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockDelta": {"delta": {"text": "Hello"}}}
            yield {"contentBlockDelta": {"delta": {"text": " world"}}}
            yield {
                "metadata": {
                    "usage": {
                        "inputTokens": 25,
                        "outputTokens": 10,
                        "cacheReadInputTokens": 0,
                    }
                }
            }

        return {"stream": stream_iter()}

    async def test_yields_text_chunks(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse_stream.return_value = await self._stream_response()

        stream = await llm.get_response_stream("Hello")
        chunks = [chunk async for chunk in stream]

        assert chunks == ["Hello", " world"]

    async def test_usage_populated_after_iteration(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse_stream.return_value = await self._stream_response()

        stream = await llm.get_response_stream("Hello")
        async for _ in stream:
            pass

        assert stream.usage is not None
        assert stream.usage.input_tokens == 25
        assert stream.usage.output_tokens == 10

    async def test_collect_returns_llm_response(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse_stream.return_value = await self._stream_response()

        stream = await llm.get_response_stream("Hello")
        response = await stream.collect()

        assert response.content == "Hello world"
        assert response.input_tokens == 25
