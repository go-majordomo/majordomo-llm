"""Tests for the Cohere provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from majordomo_llm.providers import Cohere
from majordomo_llm.base import TOKENS_PER_MILLION
from majordomo_llm.exceptions import ConfigurationError


class CountryInfo(BaseModel):
    """Test model for structured responses."""

    name: str
    capital: str
    population: int


@pytest.fixture
def mock_cohere_text_response():
    """Mock Cohere text response."""
    response = MagicMock()
    response.message.content = [MagicMock()]
    response.message.content[0].text = "Cohere says hello!"
    response.usage.tokens.input_tokens = 20
    response.usage.tokens.output_tokens = 8
    return response


@pytest.fixture
def mock_cohere_json_response():
    """Mock Cohere JSON response."""
    response = MagicMock()
    response.message.content = [MagicMock()]
    response.message.content[0].text = '{"name": "France", "capital": "Paris", "population": 67000000}'
    response.usage.tokens.input_tokens = 50
    response.usage.tokens.output_tokens = 30
    return response


class TestCohereGetResponse:
    """Tests for Cohere.get_response method."""

    @pytest.fixture
    def cohere_llm(self):
        """Create Cohere instance with mocked client."""
        with patch("majordomo_llm.providers.cohere.cohere.AsyncClientV2"):
            llm = Cohere(
                model="command-a-03-2025",
                input_cost=2.50,
                output_cost=10.00,
                api_key="test-key",
            )
            return llm

    async def test_returns_text_content(self, cohere_llm, mock_cohere_text_response):
        """Should extract text content from response."""
        cohere_llm.client.chat = AsyncMock(
            return_value=mock_cohere_text_response
        )

        response = await cohere_llm.get_response("Say hello")

        assert response.content == "Cohere says hello!"

    async def test_returns_correct_token_counts(self, cohere_llm, mock_cohere_text_response):
        """Should return correct token counts."""
        cohere_llm.client.chat = AsyncMock(
            return_value=mock_cohere_text_response
        )

        response = await cohere_llm.get_response("Test prompt")

        assert response.input_tokens == 20
        assert response.output_tokens == 8
        assert response.cached_tokens == 0

    async def test_calculates_costs_correctly(self, cohere_llm, mock_cohere_text_response):
        """Should calculate costs based on token counts and rates."""
        cohere_llm.client.chat = AsyncMock(
            return_value=mock_cohere_text_response
        )

        response = await cohere_llm.get_response("Test prompt")

        expected_input_cost = 20 * 2.50 / TOKENS_PER_MILLION
        expected_output_cost = 8 * 10.00 / TOKENS_PER_MILLION

        assert response.input_cost == expected_input_cost
        assert response.output_cost == expected_output_cost
        assert response.total_cost == expected_input_cost + expected_output_cost

    async def test_passes_temperature_and_top_p(self, cohere_llm, mock_cohere_text_response):
        """Should pass temperature and top_p to API."""
        cohere_llm.client.chat = AsyncMock(
            return_value=mock_cohere_text_response
        )

        await cohere_llm.get_response(
            "Test prompt",
            temperature=0.8,
            top_p=0.95,
        )

        call_kwargs = cohere_llm.client.chat.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["p"] == 0.95

    async def test_includes_system_prompt_in_messages(self, cohere_llm, mock_cohere_text_response):
        """Should include system prompt in messages."""
        cohere_llm.client.chat = AsyncMock(
            return_value=mock_cohere_text_response
        )

        await cohere_llm.get_response(
            "Test prompt",
            system_prompt="You are a helpful assistant.",
        )

        call_kwargs = cohere_llm.client.chat.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"


class TestCohereGetJSONResponse:
    """Tests for Cohere.get_json_response method."""

    @pytest.fixture
    def cohere_llm(self):
        """Create Cohere instance with mocked client."""
        with patch("majordomo_llm.providers.cohere.cohere.AsyncClientV2"):
            llm = Cohere(
                model="command-a-03-2025",
                input_cost=2.50,
                output_cost=10.00,
                api_key="test-key",
            )
            return llm

    async def test_parses_json_response(self, cohere_llm):
        """Should parse JSON from response text."""
        mock_response = MagicMock()
        mock_response.message.content = [MagicMock()]
        mock_response.message.content[0].text = '{"name": "test", "value": 123}'
        mock_response.usage.tokens.input_tokens = 15
        mock_response.usage.tokens.output_tokens = 10

        cohere_llm.client.chat = AsyncMock(return_value=mock_response)

        response = await cohere_llm.get_json_response("Return JSON")

        assert response.content == {"name": "test", "value": 123}

    async def test_strips_markdown_fences(self, cohere_llm):
        """Should strip markdown code fences from JSON."""
        mock_response = MagicMock()
        mock_response.message.content = [MagicMock()]
        mock_response.message.content[0].text = '```json\n{"key": "value"}\n```'
        mock_response.usage.tokens.input_tokens = 15
        mock_response.usage.tokens.output_tokens = 10

        cohere_llm.client.chat = AsyncMock(return_value=mock_response)

        response = await cohere_llm.get_json_response("Return JSON")

        assert response.content == {"key": "value"}


class TestCohereStructuredResponse:
    """Tests for Cohere structured response methods."""

    @pytest.fixture
    def cohere_llm(self):
        """Create Cohere instance with mocked client."""
        with patch("majordomo_llm.providers.cohere.cohere.AsyncClientV2"):
            llm = Cohere(
                model="command-a-03-2025",
                input_cost=2.50,
                output_cost=10.00,
                api_key="test-key",
            )
            return llm

    async def test_uses_json_mode(self, cohere_llm, mock_cohere_json_response):
        """Should use JSON mode for structured output."""
        cohere_llm.client.chat = AsyncMock(
            return_value=mock_cohere_json_response
        )

        await cohere_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        call_kwargs = cohere_llm.client.chat.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        # Schema is injected into the system prompt instead
        assert "json" in call_kwargs["messages"][0]["content"].lower()

    async def test_returns_validated_pydantic_model(self, cohere_llm, mock_cohere_json_response):
        """Should return a validated Pydantic model instance."""
        cohere_llm.client.chat = AsyncMock(
            return_value=mock_cohere_json_response
        )

        response = await cohere_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        assert isinstance(response.content, CountryInfo)
        assert response.content.name == "France"
        assert response.content.capital == "Paris"


class TestCohereInit:
    """Tests for Cohere initialization."""

    def test_raises_configuration_error_without_api_key(self):
        """Should raise ConfigurationError when no API key is provided."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                Cohere(
                    model="command-a-03-2025",
                    input_cost=2.50,
                    output_cost=10.00,
                )

            assert "CO_API_KEY" in str(exc_info.value)

    def test_sets_provider_name(self):
        """Should set provider to 'cohere'."""
        with patch("majordomo_llm.providers.cohere.cohere.AsyncClientV2"):
            llm = Cohere(
                model="command-a-03-2025",
                input_cost=2.50,
                output_cost=10.00,
                api_key="test-key",
            )

            assert llm.provider == "cohere"

    def test_stores_model_and_costs(self):
        """Should store model name and cost configuration."""
        with patch("majordomo_llm.providers.cohere.cohere.AsyncClientV2"):
            llm = Cohere(
                model="command-a-03-2025",
                input_cost=2.50,
                output_cost=10.00,
                api_key="test-key",
            )

            assert llm.model == "command-a-03-2025"
            assert llm.input_cost == 2.50
            assert llm.output_cost == 10.00

    def test_configures_client_with_api_key(self):
        """Should configure Cohere client with API key."""
        with patch("majordomo_llm.providers.cohere.cohere.AsyncClientV2") as mock_client:
            Cohere(
                model="command-a-03-2025",
                input_cost=2.50,
                output_cost=10.00,
                api_key="test-key",
            )

            mock_client.assert_called_once_with(api_key="test-key")

    def test_supports_temperature_by_default(self):
        """Should support temperature/top_p by default."""
        with patch("majordomo_llm.providers.cohere.cohere.AsyncClientV2"):
            llm = Cohere(
                model="command-a-03-2025",
                input_cost=2.50,
                output_cost=10.00,
                api_key="test-key",
            )

            assert llm.supports_temperature_top_p is True
