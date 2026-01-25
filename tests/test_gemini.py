"""Tests for the Gemini provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from majordomo_llm.providers import Gemini
from majordomo_llm.base import TOKENS_PER_MILLION
from majordomo_llm.exceptions import ConfigurationError


class CountryInfo(BaseModel):
    """Test model for structured responses."""

    name: str
    capital: str
    population: int


class TestGeminiGetResponse:
    """Tests for Gemini.get_response method."""

    @pytest.fixture
    def gemini_llm(self):
        """Create Gemini instance with mocked client."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                api_key="test-key",
            )
            return llm

    async def test_returns_text_content(self, gemini_llm, mock_gemini_text_response):
        """Should extract text content from response."""
        gemini_llm.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_text_response
        )

        response = await gemini_llm.get_response("Say hello")

        assert response.content == "Gemini says hello!"

    async def test_returns_correct_token_counts(self, gemini_llm, mock_gemini_text_response):
        """Should return correct token counts."""
        gemini_llm.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_text_response
        )

        response = await gemini_llm.get_response("Test prompt")

        assert response.input_tokens == 15
        assert response.output_tokens == 5
        assert response.cached_tokens == 0  # Gemini doesn't report cached tokens

    async def test_calculates_costs_correctly(self, gemini_llm, mock_gemini_text_response):
        """Should calculate costs based on token counts and rates."""
        gemini_llm.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_text_response
        )

        response = await gemini_llm.get_response("Test prompt")

        expected_input_cost = 15 * 0.30 / TOKENS_PER_MILLION
        expected_output_cost = 5 * 2.50 / TOKENS_PER_MILLION

        assert response.input_cost == expected_input_cost
        assert response.output_cost == expected_output_cost
        assert response.total_cost == expected_input_cost + expected_output_cost

    async def test_passes_config_parameters(self, gemini_llm, mock_gemini_text_response):
        """Should pass temperature and top_p in config."""
        gemini_llm.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_text_response
        )

        await gemini_llm.get_response(
            "Test prompt",
            system_prompt="Be helpful",
            temperature=0.5,
            top_p=0.8,
        )

        call_kwargs = gemini_llm.client.aio.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.temperature == 0.5
        assert config.top_p == 0.8
        assert config.system_instruction == "Be helpful"


class TestGeminiGetJSONResponse:
    """Tests for Gemini.get_json_response method."""

    @pytest.fixture
    def gemini_llm(self):
        """Create Gemini instance with mocked client."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                api_key="test-key",
            )
            return llm

    async def test_parses_json_response(self, gemini_llm, mock_gemini_json_response):
        """Should parse JSON from response text."""
        gemini_llm.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_json_response
        )

        response = await gemini_llm.get_json_response("Return JSON")

        assert response.content == {"status": "success", "value": 123}

    async def test_strips_markdown_fences(self, gemini_llm):
        """Should strip markdown code fences from JSON."""
        mock_response = MagicMock()
        mock_response.text = '```json\n{"key": "value"}\n```'
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 8

        gemini_llm.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        response = await gemini_llm.get_json_response("Return JSON")

        assert response.content == {"key": "value"}


class TestGeminiStructuredResponse:
    """Tests for Gemini structured response methods."""

    @pytest.fixture
    def gemini_llm(self):
        """Create Gemini instance with mocked client."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                api_key="test-key",
            )
            return llm

    async def test_passes_response_schema(self, gemini_llm):
        """Should pass JSON schema as response_schema in config."""
        mock_response = MagicMock()
        mock_response.text = '{"name": "Germany", "capital": "Berlin", "population": 83000000}'
        mock_response.usage_metadata.prompt_token_count = 30
        mock_response.usage_metadata.candidates_token_count = 20

        gemini_llm.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await gemini_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about Germany",
        )

        call_kwargs = gemini_llm.client.aio.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.response_schema is not None
        assert config.response_mime_type == "application/json"

    async def test_returns_validated_pydantic_model(self, gemini_llm):
        """Should return a validated Pydantic model instance."""
        mock_response = MagicMock()
        mock_response.text = '{"name": "Germany", "capital": "Berlin", "population": 83000000}'
        mock_response.usage_metadata.prompt_token_count = 30
        mock_response.usage_metadata.candidates_token_count = 20

        gemini_llm.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        response = await gemini_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about Germany",
        )

        assert isinstance(response.content, CountryInfo)
        assert response.content.name == "Germany"
        assert response.content.capital == "Berlin"


class TestGeminiInit:
    """Tests for Gemini initialization."""

    def test_raises_configuration_error_without_api_key(self):
        """Should raise ConfigurationError when no API key is provided."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                Gemini(
                    model="gemini-2.5-flash",
                    input_cost=0.30,
                    output_cost=2.50,
                )

            assert "GEMINI_API_KEY" in str(exc_info.value)

    def test_sets_provider_name(self):
        """Should set provider to 'gemini'."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                api_key="test-key",
            )

            assert llm.provider == "gemini"

    def test_stores_model_and_costs(self):
        """Should store model name and cost configuration."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                api_key="test-key",
            )

            assert llm.model == "gemini-2.5-flash"
            assert llm.input_cost == 0.30
            assert llm.output_cost == 2.50

    def test_always_supports_temperature(self):
        """Should always support temperature/top_p."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                api_key="test-key",
            )

            assert llm.supports_temperature_top_p is True
