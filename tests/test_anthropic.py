"""Tests for the Anthropic provider."""

import pytest
from unittest.mock import AsyncMock, patch
from pydantic import BaseModel

from majordomo_llm.providers import Anthropic
from majordomo_llm.base import TOKENS_PER_MILLION
from majordomo_llm.exceptions import ConfigurationError


class CountryInfo(BaseModel):
    """Test model for structured responses."""

    name: str
    capital: str
    population: int


class TestAnthropicGetResponse:
    """Tests for Anthropic.get_response method."""

    @pytest.fixture
    def anthropic_llm(self):
        """Create Anthropic instance with mocked client."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-4-20250514",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )
            return llm

    async def test_returns_text_content(self, anthropic_llm, mock_anthropic_text_response):
        """Should extract text content from response."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        response = await anthropic_llm.get_response("What is the capital of France?")

        assert response.content == "Paris is the capital of France."

    async def test_returns_correct_token_counts(self, anthropic_llm, mock_anthropic_text_response):
        """Should return correct token counts."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        response = await anthropic_llm.get_response("Test prompt")

        assert response.input_tokens == 25
        assert response.output_tokens == 10
        assert response.cached_tokens == 0

    async def test_calculates_costs_correctly(self, anthropic_llm, mock_anthropic_text_response):
        """Should calculate costs based on token counts and rates."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        response = await anthropic_llm.get_response("Test prompt")

        expected_input_cost = 25 * 3.0 / TOKENS_PER_MILLION
        expected_output_cost = 10 * 15.0 / TOKENS_PER_MILLION

        assert response.input_cost == expected_input_cost
        assert response.output_cost == expected_output_cost
        assert response.total_cost == expected_input_cost + expected_output_cost

    async def test_includes_response_time(self, anthropic_llm, mock_anthropic_text_response):
        """Should include response time measurement."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        response = await anthropic_llm.get_response("Test prompt")

        assert response.response_time >= 0

    async def test_passes_temperature_and_top_p(self, anthropic_llm, mock_anthropic_text_response):
        """Should pass temperature and top_p to API."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        await anthropic_llm.get_response(
            "Test prompt",
            temperature=0.7,
            top_p=0.9,
        )

        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9

    async def test_uses_default_system_prompt(self, anthropic_llm, mock_anthropic_text_response):
        """Should use default system prompt when none provided."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        await anthropic_llm.get_response("Test prompt")

        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        system_text = call_kwargs["system"][0]["text"]
        assert "helpful assistant" in system_text


class TestAnthropicStructuredResponse:
    """Tests for Anthropic structured response methods."""

    @pytest.fixture
    def anthropic_llm(self):
        """Create Anthropic instance with mocked client."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-4-20250514",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )
            return llm

    async def test_extracts_tool_use_content(self, anthropic_llm, mock_anthropic_tool_response):
        """Should extract content from tool_use block."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_tool_response)

        response = await anthropic_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        assert response.content.name == "France"
        assert response.content.capital == "Paris"
        assert response.content.population == 67000000

    async def test_returns_validated_pydantic_model(self, anthropic_llm, mock_anthropic_tool_response):
        """Should return a validated Pydantic model instance."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_tool_response)

        response = await anthropic_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        assert isinstance(response.content, CountryInfo)

    async def test_forces_tool_choice(self, anthropic_llm, mock_anthropic_tool_response):
        """Should force tool choice to structured_response."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_tool_response)

        await anthropic_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"]["type"] == "tool"
        assert call_kwargs["tool_choice"]["name"] == "structured_response"


class TestAnthropicInit:
    """Tests for Anthropic initialization."""

    def test_raises_configuration_error_without_api_key(self):
        """Should raise ConfigurationError when no API key is provided."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                Anthropic(
                    model="claude-sonnet-4-20250514",
                    input_cost=3.0,
                    output_cost=15.0,
                )

            assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_sets_provider_name(self):
        """Should set provider to 'anthropic'."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-4-20250514",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )

            assert llm.provider == "anthropic"

    def test_stores_model_and_costs(self):
        """Should store model name and cost configuration."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-4-20250514",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )

            assert llm.model == "claude-sonnet-4-20250514"
            assert llm.input_cost == 3.0
            assert llm.output_cost == 15.0

    def test_web_search_disabled_by_default(self):
        """Should have web search disabled by default."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-4-20250514",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )

            assert llm.use_web_search is False

    def test_web_search_can_be_enabled(self):
        """Should allow enabling web search."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-4-5-20250929",
                input_cost=3.0,
                output_cost=15.0,
                use_web_search=True,
                api_key="test-key",
            )

            assert llm.use_web_search is True
