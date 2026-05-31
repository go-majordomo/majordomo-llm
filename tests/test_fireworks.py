"""Tests for the Fireworks provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from majordomo_llm.base import TOKENS_PER_MILLION
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers import Fireworks

MODEL_ID = "accounts/fireworks/models/deepseek-v4-pro"


class CountryInfo(BaseModel):
    """Test model for structured responses."""

    name: str
    capital: str
    population: int


COUNTRY_SCHEMA = CountryInfo.model_json_schema()


@pytest.fixture
def mock_fireworks_text_response():
    """Mock Fireworks text response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Fireworks says hello!"
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 8
    response.usage.prompt_tokens_details = None
    return response


@pytest.fixture
def mock_fireworks_json_response():
    """Mock Fireworks JSON response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[
        0
    ].message.content = '{"name": "France", "capital": "Paris", "population": 67000000}'
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 30
    response.usage.prompt_tokens_details = None
    return response


class TestFireworksGetResponse:
    """Tests for Fireworks.get_response method."""

    @pytest.fixture
    def fireworks_llm(self):
        """Create Fireworks instance with mocked client."""
        with patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"):
            llm = Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
            )
            return llm

    async def test_returns_text_content(self, fireworks_llm, mock_fireworks_text_response):
        """Should extract text content from response."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_text_response
        )

        response = await fireworks_llm.get_response("Say hello")

        assert response.content == "Fireworks says hello!"

    async def test_returns_correct_token_counts(
        self, fireworks_llm, mock_fireworks_text_response
    ):
        """Should return correct token counts."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_text_response
        )

        response = await fireworks_llm.get_response("Test prompt")

        assert response.input_tokens == 20
        assert response.output_tokens == 8
        assert response.cached_tokens == 0

    async def test_calculates_costs_correctly(self, fireworks_llm, mock_fireworks_text_response):
        """Should calculate costs based on token counts and rates."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_text_response
        )

        response = await fireworks_llm.get_response("Test prompt")

        expected_input_cost = 20 * 1.74 / TOKENS_PER_MILLION
        expected_output_cost = 8 * 3.48 / TOKENS_PER_MILLION

        assert response.input_cost == expected_input_cost
        assert response.output_cost == expected_output_cost
        assert response.total_cost == expected_input_cost + expected_output_cost

    async def test_passes_temperature_and_top_p(
        self, fireworks_llm, mock_fireworks_text_response
    ):
        """Should pass temperature and top_p to API."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_text_response
        )

        await fireworks_llm.get_response(
            "Test prompt",
            temperature=0.8,
            top_p=0.95,
        )

        call_kwargs = fireworks_llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.95

    async def test_includes_system_prompt_in_messages(
        self, fireworks_llm, mock_fireworks_text_response
    ):
        """Should include system prompt in messages."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_text_response
        )

        await fireworks_llm.get_response(
            "Test prompt",
            system_prompt="You are a helpful assistant.",
        )

        call_kwargs = fireworks_llm.client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"

    async def test_collapses_reasoning_effort_and_thinking_enabled(
        self, mock_fireworks_text_response
    ):
        """Fireworks rejects requests with both reasoning_effort and thinking
        set ('cannot specify both...'). When the user sets both with thinking
        enabled, we send only reasoning_effort (which implies thinking is on)."""
        with patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"):
            llm = Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
                reasoning_effort="medium",
                thinking="enabled",
            )
        llm.client.chat.completions.create = AsyncMock(return_value=mock_fireworks_text_response)

        await llm.get_response("Test prompt")

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "medium"
        assert "extra_body" not in call_kwargs

    async def test_thinking_disabled_wins_over_reasoning_effort(
        self, mock_fireworks_text_response
    ):
        """Explicit thinking=disabled is authoritative — drop reasoning_effort
        so Fireworks doesn't reject the request."""
        with patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"):
            llm = Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
                reasoning_effort="medium",
                thinking="disabled",
            )
        llm.client.chat.completions.create = AsyncMock(return_value=mock_fireworks_text_response)

        await llm.get_response("Test prompt")

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs
        assert call_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}

    async def test_reasoning_effort_alone(self, mock_fireworks_text_response):
        """reasoning_effort without thinking → sent as-is."""
        with patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"):
            llm = Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
                reasoning_effort="high",
            )
        llm.client.chat.completions.create = AsyncMock(return_value=mock_fireworks_text_response)

        await llm.get_response("Test prompt")

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "high"
        assert "extra_body" not in call_kwargs

    async def test_omits_reasoning_options_when_unset(
        self, fireworks_llm, mock_fireworks_text_response
    ):
        """Should not pass reasoning_effort or extra_body when neither is set."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_text_response
        )

        await fireworks_llm.get_response("Test prompt")

        call_kwargs = fireworks_llm.client.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs
        assert "extra_body" not in call_kwargs


class TestFireworksGetJSONResponse:
    """Tests for Fireworks.get_json_response method."""

    @pytest.fixture
    def fireworks_llm(self):
        """Create Fireworks instance with mocked client."""
        with patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"):
            llm = Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
            )
            return llm

    async def test_parses_json_response(self, fireworks_llm):
        """Should parse JSON from response text."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "test", "value": 123}'
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 10
        mock_response.usage.prompt_tokens_details = None

        fireworks_llm.client.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await fireworks_llm.get_json_response("Return JSON")

        assert response.content == {"name": "test", "value": 123}

    async def test_strips_markdown_fences(self, fireworks_llm):
        """Should strip markdown code fences from JSON."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"key": "value"}\n```'
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 10
        mock_response.usage.prompt_tokens_details = None

        fireworks_llm.client.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await fireworks_llm.get_json_response("Return JSON")

        assert response.content == {"key": "value"}


class TestFireworksStructuredResponse:
    """Tests for Fireworks structured response methods."""

    @pytest.fixture
    def fireworks_llm(self):
        """Create Fireworks instance with mocked client."""
        with patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"):
            llm = Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
            )
            return llm

    async def test_uses_json_schema_response_format(
        self, fireworks_llm, mock_fireworks_json_response
    ):
        """Should use JSON schema response format for structured output."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_json_response
        )

        await fireworks_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        call_kwargs = fireworks_llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "description": "Provide a structured response using the CountryInfo schema",
                "name": "CountryInfo",
                "schema": COUNTRY_SCHEMA,
                "strict": True,
            },
        }

    async def test_returns_validated_pydantic_model(
        self, fireworks_llm, mock_fireworks_json_response
    ):
        """Should return a validated Pydantic model instance."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_json_response
        )

        response = await fireworks_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        assert isinstance(response.content, CountryInfo)
        assert response.content.name == "France"
        assert response.content.capital == "Paris"

    async def test_json_schema_response_returns_canonical_json(
        self, fireworks_llm, mock_fireworks_json_response
    ):
        """Should return canonical JSON string for raw schema calls."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_json_response
        )

        response = await fireworks_llm.get_json_schema_response(
            user_prompt="Tell me about France",
            response_schema=COUNTRY_SCHEMA,
            schema_name="CountryInfo",
        )

        assert response.content == '{"capital":"Paris","name":"France","population":67000000}'


class TestFireworksInit:
    """Tests for Fireworks initialization."""

    def test_raises_configuration_error_without_api_key(self):
        """Should raise ConfigurationError when no API key is provided."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                Fireworks(
                    model=MODEL_ID,
                    input_cost=1.74,
                    output_cost=3.48,
                )

            assert "FIREWORKS_API_KEY" in str(exc_info.value)

    def test_sets_provider_name(self):
        """Should set provider to 'fireworks'."""
        with patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"):
            llm = Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
            )

            assert llm.provider == "fireworks"

    def test_stores_model_and_costs(self):
        """Should store model name and cost configuration."""
        with patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"):
            llm = Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
            )

            assert llm.model == MODEL_ID
            assert llm.input_cost == 1.74
            assert llm.output_cost == 3.48

    def test_configures_client_with_fireworks_base_url(self):
        """Should configure OpenAI client with Fireworks base URL."""
        with patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI") as mock_client:
            Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
            )

            mock_client.assert_called_once_with(
                api_key="test-key",
                base_url="https://api.fireworks.ai/inference/v1",
                default_headers=None,
            )

    def test_accepts_custom_base_url_and_headers(self):
        """Should honor explicit base_url and default_headers overrides."""
        with patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI") as mock_client:
            Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
                api_key_alias="prod-pool",
                base_url="https://custom.example.com/v1",
                default_headers={"X-Trace": "abc"},
            )

            mock_client.assert_called_once_with(
                api_key="test-key",
                base_url="https://custom.example.com/v1",
                default_headers={"X-Trace": "abc"},
            )

    def test_supports_temperature_by_default(self):
        """Should support temperature/top_p by default."""
        with patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"):
            llm = Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
            )

            assert llm.supports_temperature_top_p is True

    def test_sets_reasoning_options(self):
        """Should store reasoning_effort and thinking when provided."""
        with patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"):
            llm = Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
                reasoning_effort="high",
                thinking="enabled",
            )

            assert llm.reasoning_effort == "high"
            assert llm.thinking == "enabled"

    @pytest.mark.parametrize(
        ("reasoning_effort", "thinking", "error_match"),
        [
            ("extreme", "disabled", "Invalid Fireworks reasoning_effort"),
            ("medium", "maybe", "Invalid Fireworks thinking mode"),
        ],
    )
    def test_rejects_invalid_reasoning_options(self, reasoning_effort, thinking, error_match):
        """Should reject invalid reasoning configuration."""
        with (
            patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"),
            pytest.raises(ValueError, match=error_match),
        ):
            Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
                reasoning_effort=reasoning_effort,
                thinking=thinking,
            )


class TestFireworksGetResponseStream:
    """Tests for Fireworks.get_response_stream method."""

    @pytest.fixture
    def fireworks_llm(self):
        """Create Fireworks instance with mocked client."""
        with patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"):
            llm = Fireworks(
                model=MODEL_ID,
                input_cost=1.74,
                output_cost=3.48,
                api_key="test-key",
            )
            return llm

    async def test_yields_text_chunks(self, fireworks_llm, mock_fireworks_stream_chunks):
        """Should yield text chunks from stream."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_stream_chunks
        )

        stream = await fireworks_llm.get_response_stream("Hello")
        chunks = [chunk async for chunk in stream]

        assert chunks == ["Hello", " world"]

    async def test_usage_populated_after_iteration(
        self, fireworks_llm, mock_fireworks_stream_chunks
    ):
        """Should populate usage after stream is consumed."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_stream_chunks
        )

        stream = await fireworks_llm.get_response_stream("Hello")
        async for _ in stream:
            pass

        assert stream.usage is not None
        assert stream.usage.input_tokens == 20
        assert stream.usage.output_tokens == 8

    async def test_calculates_costs_correctly(
        self, fireworks_llm, mock_fireworks_stream_chunks
    ):
        """Should calculate costs from stream usage."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_stream_chunks
        )

        stream = await fireworks_llm.get_response_stream("Hello")
        async for _ in stream:
            pass

        expected_input_cost = 20 * 1.74 / TOKENS_PER_MILLION
        expected_output_cost = 8 * 3.48 / TOKENS_PER_MILLION
        assert stream.usage.input_cost == expected_input_cost
        assert stream.usage.output_cost == expected_output_cost

    async def test_collect_returns_llm_response(
        self, fireworks_llm, mock_fireworks_stream_chunks
    ):
        """Should return LLMResponse from collect()."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_stream_chunks
        )

        stream = await fireworks_llm.get_response_stream("Hello")
        response = await stream.collect()

        assert response.content == "Hello world"
        assert response.input_tokens == 20

    async def test_passes_stream_options_to_api(
        self, fireworks_llm, mock_fireworks_stream_chunks
    ):
        """Should pass stream=True and stream_options to API."""
        fireworks_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_fireworks_stream_chunks
        )

        await fireworks_llm.get_response_stream("Hello")

        call_kwargs = fireworks_llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}
