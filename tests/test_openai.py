"""Tests for the OpenAI provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from majordomo_llm.providers import OpenAI
from majordomo_llm.base import TOKENS_PER_MILLION
from majordomo_llm.exceptions import ConfigurationError


class CountryInfo(BaseModel):
    """Test model for structured responses."""

    name: str
    capital: str
    population: int


class TestOpenAIGetResponse:
    """Tests for OpenAI.get_response method."""

    @pytest.fixture
    def openai_llm(self):
        """Create OpenAI instance with mocked client."""
        with patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"):
            llm = OpenAI(
                model="gpt-4o",
                input_cost=2.5,
                output_cost=10.0,
                api_key="test-key",
            )
            return llm

    async def test_returns_text_content(self, openai_llm, mock_openai_text_response):
        """Should extract text content from response."""
        openai_llm.client.responses.create = AsyncMock(return_value=mock_openai_text_response)

        response = await openai_llm.get_response("What is the meaning of life?")

        assert response.content == "The answer is 42."

    async def test_returns_correct_token_counts(self, openai_llm, mock_openai_text_response):
        """Should return correct token counts."""
        openai_llm.client.responses.create = AsyncMock(return_value=mock_openai_text_response)

        response = await openai_llm.get_response("Test prompt")

        assert response.input_tokens == 20
        assert response.output_tokens == 8
        assert response.cached_tokens == 0

    async def test_calculates_costs_correctly(self, openai_llm, mock_openai_text_response):
        """Should calculate costs based on token counts and rates."""
        openai_llm.client.responses.create = AsyncMock(return_value=mock_openai_text_response)

        response = await openai_llm.get_response("Test prompt")

        expected_input_cost = 20 * 2.5 / TOKENS_PER_MILLION
        expected_output_cost = 8 * 10.0 / TOKENS_PER_MILLION

        assert response.input_cost == expected_input_cost
        assert response.output_cost == expected_output_cost
        assert response.total_cost == expected_input_cost + expected_output_cost

    async def test_passes_temperature_and_top_p(self, openai_llm, mock_openai_text_response):
        """Should pass temperature and top_p to API."""
        openai_llm.client.responses.create = AsyncMock(return_value=mock_openai_text_response)

        await openai_llm.get_response(
            "Test prompt",
            temperature=0.8,
            top_p=0.95,
        )

        call_kwargs = openai_llm.client.responses.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.95


class TestOpenAIGetJSONResponse:
    """Tests for OpenAI.get_json_response method."""

    @pytest.fixture
    def openai_llm(self):
        """Create OpenAI instance with mocked client."""
        with patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"):
            llm = OpenAI(
                model="gpt-4o",
                input_cost=2.5,
                output_cost=10.0,
                api_key="test-key",
            )
            return llm

    async def test_parses_json_response(self, openai_llm):
        """Should parse JSON from response text."""
        mock_response = MagicMock()
        mock_response.output_text = '{"name": "test", "value": 123}'
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 10
        mock_response.usage.input_tokens_details.cached_tokens = 0

        openai_llm.client.responses.create = AsyncMock(return_value=mock_response)

        response = await openai_llm.get_json_response("Return JSON")

        assert response.content == {"name": "test", "value": 123}

    async def test_strips_markdown_fences(self, openai_llm):
        """Should strip markdown code fences from JSON."""
        mock_response = MagicMock()
        mock_response.output_text = '```json\n{"key": "value"}\n```'
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 10
        mock_response.usage.input_tokens_details.cached_tokens = 0

        openai_llm.client.responses.create = AsyncMock(return_value=mock_response)

        response = await openai_llm.get_json_response("Return JSON")

        assert response.content == {"key": "value"}


class TestOpenAIStructuredResponse:
    """Tests for OpenAI structured response methods."""

    @pytest.fixture
    def openai_llm(self):
        """Create OpenAI instance with mocked client."""
        with patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"):
            llm = OpenAI(
                model="gpt-4o",
                input_cost=2.5,
                output_cost=10.0,
                api_key="test-key",
            )
            return llm

    async def test_uses_parse_endpoint(self, openai_llm, mock_openai_structured_response):
        """Should use responses.parse for structured output."""
        openai_llm.client.responses.parse = AsyncMock(return_value=mock_openai_structured_response)

        await openai_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about Japan",
        )

        openai_llm.client.responses.parse.assert_called_once()

    async def test_passes_response_model_as_text_format(self, openai_llm, mock_openai_structured_response):
        """Should pass Pydantic model as text_format."""
        openai_llm.client.responses.parse = AsyncMock(return_value=mock_openai_structured_response)

        await openai_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about Japan",
        )

        call_kwargs = openai_llm.client.responses.parse.call_args.kwargs
        assert call_kwargs["text_format"] == CountryInfo

    async def test_returns_validated_pydantic_model(self, openai_llm, mock_openai_structured_response):
        """Should return a validated Pydantic model instance."""
        openai_llm.client.responses.parse = AsyncMock(return_value=mock_openai_structured_response)

        response = await openai_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about Japan",
        )

        assert isinstance(response.content, CountryInfo)
        assert response.content.name == "Japan"
        assert response.content.capital == "Tokyo"


class TestOpenAIInit:
    """Tests for OpenAI initialization."""

    def test_raises_configuration_error_without_api_key(self):
        """Should raise ConfigurationError when no API key is provided."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                OpenAI(
                    model="gpt-4o",
                    input_cost=2.5,
                    output_cost=10.0,
                )

            assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_sets_provider_name(self):
        """Should set provider to 'openai'."""
        with patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"):
            llm = OpenAI(
                model="gpt-4o",
                input_cost=2.5,
                output_cost=10.0,
                api_key="test-key",
            )

            assert llm.provider == "openai"

    def test_stores_model_and_costs(self):
        """Should store model name and cost configuration."""
        with patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"):
            llm = OpenAI(
                model="gpt-4o",
                input_cost=2.5,
                output_cost=10.0,
                api_key="test-key",
            )

            assert llm.model == "gpt-4o"
            assert llm.input_cost == 2.5
            assert llm.output_cost == 10.0

    def test_supports_temperature_by_default(self):
        """Should support temperature/top_p by default."""
        with patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"):
            llm = OpenAI(
                model="gpt-4o",
                input_cost=2.5,
                output_cost=10.0,
                api_key="test-key",
            )

            assert llm.supports_temperature_top_p is True

    def test_can_disable_temperature_support(self):
        """Should allow disabling temperature/top_p support."""
        with patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"):
            llm = OpenAI(
                model="gpt-5",
                input_cost=1.25,
                output_cost=10.0,
                supports_temperature_top_p=False,
                api_key="test-key",
            )

            assert llm.supports_temperature_top_p is False
