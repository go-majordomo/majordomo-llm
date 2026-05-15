"""Tests for the DeepSeek provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from majordomo_llm.base import TOKENS_PER_MILLION
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers import DeepSeek


class CountryInfo(BaseModel):
    """Test model for structured responses."""

    name: str
    capital: str
    population: int


@pytest.fixture
def mock_deepseek_text_response():
    """Mock DeepSeek text response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "DeepSeek says hello!"
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 8
    response.usage.prompt_tokens_details = None
    return response


@pytest.fixture
def mock_deepseek_json_response():
    """Mock DeepSeek JSON response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[
        0
    ].message.content = '{"name": "France", "capital": "Paris", "population": 67000000}'
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 30
    response.usage.prompt_tokens_details = None
    return response


class TestDeepSeekGetResponse:
    """Tests for DeepSeek.get_response method."""

    @pytest.fixture
    def deepseek_llm(self):
        """Create DeepSeek instance with mocked client."""
        with patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"):
            llm = DeepSeek(
                model="deepseek-chat",
                input_cost=0.28,
                output_cost=0.42,
                api_key="test-key",
            )
            return llm

    async def test_returns_text_content(self, deepseek_llm, mock_deepseek_text_response):
        """Should extract text content from response."""
        deepseek_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_deepseek_text_response
        )

        response = await deepseek_llm.get_response("Say hello")

        assert response.content == "DeepSeek says hello!"

    async def test_returns_correct_token_counts(self, deepseek_llm, mock_deepseek_text_response):
        """Should return correct token counts."""
        deepseek_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_deepseek_text_response
        )

        response = await deepseek_llm.get_response("Test prompt")

        assert response.input_tokens == 20
        assert response.output_tokens == 8
        assert response.cached_tokens == 0

    async def test_calculates_costs_correctly(self, deepseek_llm, mock_deepseek_text_response):
        """Should calculate costs based on token counts and rates."""
        deepseek_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_deepseek_text_response
        )

        response = await deepseek_llm.get_response("Test prompt")

        expected_input_cost = 20 * 0.28 / TOKENS_PER_MILLION
        expected_output_cost = 8 * 0.42 / TOKENS_PER_MILLION

        assert response.input_cost == expected_input_cost
        assert response.output_cost == expected_output_cost
        assert response.total_cost == expected_input_cost + expected_output_cost

    async def test_passes_temperature_and_top_p(self, deepseek_llm, mock_deepseek_text_response):
        """Should pass temperature and top_p to API."""
        deepseek_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_deepseek_text_response
        )

        await deepseek_llm.get_response(
            "Test prompt",
            temperature=0.8,
            top_p=0.95,
        )

        call_kwargs = deepseek_llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.95

    async def test_includes_system_prompt_in_messages(
        self, deepseek_llm, mock_deepseek_text_response
    ):
        """Should include system prompt in messages."""
        deepseek_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_deepseek_text_response
        )

        await deepseek_llm.get_response(
            "Test prompt",
            system_prompt="You are a helpful assistant.",
        )

        call_kwargs = deepseek_llm.client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"

    async def test_passes_deepseek_reasoning_options(self, mock_deepseek_text_response):
        """Should pass configured DeepSeek reasoning options to chat completions."""
        with patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"):
            llm = DeepSeek(
                model="deepseek-v4-pro",
                input_cost=0.435,
                output_cost=0.87,
                api_key="test-key",
                reasoning_effort="medium",
                thinking="disabled",
            )
        llm.client.chat.completions.create = AsyncMock(return_value=mock_deepseek_text_response)

        await llm.get_response("Test prompt")

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "medium"
        assert call_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


class TestDeepSeekGetJSONResponse:
    """Tests for DeepSeek.get_json_response method."""

    @pytest.fixture
    def deepseek_llm(self):
        """Create DeepSeek instance with mocked client."""
        with patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"):
            llm = DeepSeek(
                model="deepseek-chat",
                input_cost=0.28,
                output_cost=0.42,
                api_key="test-key",
            )
            return llm

    async def test_parses_json_response(self, deepseek_llm):
        """Should parse JSON from response text."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "test", "value": 123}'
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 10
        mock_response.usage.prompt_tokens_details = None

        deepseek_llm.client.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await deepseek_llm.get_json_response("Return JSON")

        assert response.content == {"name": "test", "value": 123}

    async def test_strips_markdown_fences(self, deepseek_llm):
        """Should strip markdown code fences from JSON."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"key": "value"}\n```'
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 10
        mock_response.usage.prompt_tokens_details = None

        deepseek_llm.client.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await deepseek_llm.get_json_response("Return JSON")

        assert response.content == {"key": "value"}


class TestDeepSeekStructuredResponse:
    """Tests for DeepSeek structured response methods."""

    @pytest.fixture
    def deepseek_llm(self):
        """Create DeepSeek instance with mocked client."""
        with patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"):
            llm = DeepSeek(
                model="deepseek-chat",
                input_cost=0.28,
                output_cost=0.42,
                api_key="test-key",
            )
            return llm

    async def test_uses_json_mode(self, deepseek_llm, mock_deepseek_json_response):
        """Should use JSON mode for structured output."""
        deepseek_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_deepseek_json_response
        )

        await deepseek_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        call_kwargs = deepseek_llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    async def test_structured_response_passes_deepseek_reasoning_options(
        self, mock_deepseek_json_response
    ):
        """Should pass configured DeepSeek reasoning options to structured calls."""
        with patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"):
            llm = DeepSeek(
                model="deepseek-v4-flash",
                input_cost=0.14,
                output_cost=0.28,
                api_key="test-key",
                reasoning_effort="medium",
                thinking="disabled",
            )
        llm.client.chat.completions.create = AsyncMock(return_value=mock_deepseek_json_response)

        await llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "medium"
        assert call_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}

    async def test_returns_validated_pydantic_model(
        self, deepseek_llm, mock_deepseek_json_response
    ):
        """Should return a validated Pydantic model instance."""
        deepseek_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_deepseek_json_response
        )

        response = await deepseek_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        assert isinstance(response.content, CountryInfo)
        assert response.content.name == "France"
        assert response.content.capital == "Paris"


class TestDeepSeekInit:
    """Tests for DeepSeek initialization."""

    def test_raises_configuration_error_without_api_key(self):
        """Should raise ConfigurationError when no API key is provided."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                DeepSeek(
                    model="deepseek-chat",
                    input_cost=0.28,
                    output_cost=0.42,
                )

            assert "DEEPSEEK_API_KEY" in str(exc_info.value)

    def test_sets_provider_name(self):
        """Should set provider to 'deepseek'."""
        with patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"):
            llm = DeepSeek(
                model="deepseek-chat",
                input_cost=0.28,
                output_cost=0.42,
                api_key="test-key",
            )

            assert llm.provider == "deepseek"

    def test_stores_model_and_costs(self):
        """Should store model name and cost configuration."""
        with patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"):
            llm = DeepSeek(
                model="deepseek-chat",
                input_cost=0.28,
                output_cost=0.42,
                api_key="test-key",
            )

            assert llm.model == "deepseek-chat"
            assert llm.input_cost == 0.28
            assert llm.output_cost == 0.42

    def test_configures_client_with_deepseek_base_url(self):
        """Should configure OpenAI client with DeepSeek base URL."""
        with patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI") as mock_client:
            DeepSeek(
                model="deepseek-chat",
                input_cost=0.28,
                output_cost=0.42,
                api_key="test-key",
            )

            mock_client.assert_called_once_with(
                api_key="test-key",
                base_url="https://api.deepseek.com",
                default_headers=None,
            )

    def test_supports_temperature_by_default(self):
        """Should support temperature/top_p by default."""
        with patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"):
            llm = DeepSeek(
                model="deepseek-chat",
                input_cost=0.28,
                output_cost=0.42,
                api_key="test-key",
            )

            assert llm.supports_temperature_top_p is True

    def test_sets_deepseek_reasoning_options(self):
        """Should set DeepSeek reasoning options."""
        with patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"):
            llm = DeepSeek(
                model="deepseek-v4-pro",
                input_cost=0.435,
                output_cost=0.87,
                api_key="test-key",
                reasoning_effort="medium",
                thinking="disabled",
            )

            assert llm.reasoning_effort == "medium"
            assert llm.thinking == "disabled"

    @pytest.mark.parametrize(
        ("reasoning_effort", "thinking", "error_match"),
        [
            ("extreme", "disabled", "Invalid DeepSeek reasoning_effort"),
            ("medium", "maybe", "Invalid DeepSeek thinking mode"),
        ],
    )
    def test_rejects_invalid_reasoning_options(self, reasoning_effort, thinking, error_match):
        """Should reject invalid DeepSeek reasoning configuration."""
        with (
            patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"),
            pytest.raises(ValueError, match=error_match),
        ):
            DeepSeek(
                model="deepseek-v4-pro",
                input_cost=0.435,
                output_cost=0.87,
                api_key="test-key",
                reasoning_effort=reasoning_effort,
                thinking=thinking,
            )


class TestDeepSeekGetResponseStream:
    """Tests for DeepSeek.get_response_stream method."""

    @pytest.fixture
    def deepseek_llm(self):
        """Create DeepSeek instance with mocked client."""
        with patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"):
            llm = DeepSeek(
                model="deepseek-chat",
                input_cost=0.28,
                output_cost=0.42,
                api_key="test-key",
            )
            return llm

    async def test_yields_text_chunks(self, deepseek_llm, mock_deepseek_stream_chunks):
        """Should yield text chunks from stream."""
        deepseek_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_deepseek_stream_chunks
        )

        stream = await deepseek_llm.get_response_stream("Hello")
        chunks = [chunk async for chunk in stream]

        assert chunks == ["Hello", " world"]

    async def test_usage_populated_after_iteration(self, deepseek_llm, mock_deepseek_stream_chunks):
        """Should populate usage after stream is consumed."""
        deepseek_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_deepseek_stream_chunks
        )

        stream = await deepseek_llm.get_response_stream("Hello")
        async for _ in stream:
            pass

        assert stream.usage is not None
        assert stream.usage.input_tokens == 20
        assert stream.usage.output_tokens == 8

    async def test_calculates_costs_correctly(self, deepseek_llm, mock_deepseek_stream_chunks):
        """Should calculate costs from stream usage."""
        deepseek_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_deepseek_stream_chunks
        )

        stream = await deepseek_llm.get_response_stream("Hello")
        async for _ in stream:
            pass

        expected_input_cost = 20 * 0.28 / TOKENS_PER_MILLION
        expected_output_cost = 8 * 0.42 / TOKENS_PER_MILLION
        assert stream.usage.input_cost == expected_input_cost
        assert stream.usage.output_cost == expected_output_cost

    async def test_collect_returns_llm_response(self, deepseek_llm, mock_deepseek_stream_chunks):
        """Should return LLMResponse from collect()."""
        deepseek_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_deepseek_stream_chunks
        )

        stream = await deepseek_llm.get_response_stream("Hello")
        response = await stream.collect()

        assert response.content == "Hello world"
        assert response.input_tokens == 20

    async def test_passes_stream_options_to_api(self, deepseek_llm, mock_deepseek_stream_chunks):
        """Should pass stream=True and stream_options to API."""
        deepseek_llm.client.chat.completions.create = AsyncMock(
            return_value=mock_deepseek_stream_chunks
        )

        await deepseek_llm.get_response_stream("Hello")

        call_kwargs = deepseek_llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}

    async def test_stream_passes_deepseek_reasoning_options(self, mock_deepseek_stream_chunks):
        """Should pass configured DeepSeek reasoning options to streaming calls."""
        with patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"):
            llm = DeepSeek(
                model="deepseek-v4-flash",
                input_cost=0.14,
                output_cost=0.28,
                api_key="test-key",
                reasoning_effort="medium",
                thinking="disabled",
            )
        llm.client.chat.completions.create = AsyncMock(return_value=mock_deepseek_stream_chunks)

        await llm.get_response_stream("Hello")

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "medium"
        assert call_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
