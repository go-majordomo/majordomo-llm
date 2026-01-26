"""Tests for the LLMCascade class."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from majordomo_llm import LLMCascade
from majordomo_llm.exceptions import ProviderError


@pytest.fixture
def mock_all_clients():
    """Mock all provider API clients and environment variables."""
    env_vars = {
        "ANTHROPIC_API_KEY": "test-key",
        "OPENAI_API_KEY": "test-key",
        "GEMINI_API_KEY": "test-key",
        "DEEPSEEK_API_KEY": "test-key",
        "CO_API_KEY": "test-key",
    }
    with (
        patch.dict("os.environ", env_vars),
        patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"),
        patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"),
        patch("majordomo_llm.providers.gemini.genai.Client"),
        patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"),
        patch("majordomo_llm.providers.cohere.cohere.AsyncClientV2"),
    ):
        yield


class TestLLMCascadeInit:
    """Tests for LLMCascade initialization."""

    def test_creates_llm_instances_for_all_providers(self, mock_all_clients):
        """Should create LLM instances for all providers in list."""
        cascade = LLMCascade([
            ("anthropic", "claude-sonnet-4-20250514"),
            ("openai", "gpt-4o"),
        ])

        assert len(cascade.llms) == 2
        assert cascade.llms[0].provider == "anthropic"
        assert cascade.llms[1].provider == "openai"

    def test_sets_provider_to_cascade(self, mock_all_clients):
        """Should set provider name to 'cascade'."""
        cascade = LLMCascade([
            ("anthropic", "claude-sonnet-4-20250514"),
        ])

        assert cascade.provider == "cascade"

    def test_uses_primary_provider_attributes(self, mock_all_clients):
        """Should use first provider's attributes for metadata."""
        cascade = LLMCascade([
            ("anthropic", "claude-sonnet-4-20250514"),
            ("openai", "gpt-4o"),
        ])

        assert cascade.model == "claude-sonnet-4-20250514"
        assert cascade.input_cost == 3.00
        assert cascade.output_cost == 15.00

    def test_raises_error_for_empty_providers(self):
        """Should raise ValueError for empty providers list."""
        with pytest.raises(ValueError) as exc_info:
            LLMCascade([])

        assert "at least one provider" in str(exc_info.value)


class TestLLMCascadeGetResponse:
    """Tests for LLMCascade.get_response method."""

    @pytest.fixture
    def cascade(self, mock_all_clients):
        """Create LLMCascade with mocked providers."""
        return LLMCascade([
            ("anthropic", "claude-sonnet-4-20250514"),
            ("openai", "gpt-4o"),
            ("gemini", "gemini-2.5-flash"),
        ])

    async def test_returns_response_from_primary_provider(self, cascade):
        """Should return response from first provider when it succeeds."""
        mock_response = MagicMock()
        mock_response.content = "Response from Anthropic"

        cascade.llms[0].get_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_response("Test prompt")

        assert response.content == "Response from Anthropic"
        cascade.llms[0].get_response.assert_called_once()

    async def test_falls_back_to_second_provider_on_failure(self, cascade):
        """Should fall back to second provider when first fails."""
        mock_response = MagicMock()
        mock_response.content = "Response from OpenAI"

        cascade.llms[0].get_response = AsyncMock(
            side_effect=ProviderError("Anthropic down", provider="anthropic")
        )
        cascade.llms[1].get_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_response("Test prompt")

        assert response.content == "Response from OpenAI"
        cascade.llms[0].get_response.assert_called_once()
        cascade.llms[1].get_response.assert_called_once()

    async def test_falls_back_through_all_providers(self, cascade):
        """Should try all providers in order until one succeeds."""
        mock_response = MagicMock()
        mock_response.content = "Response from Gemini"

        cascade.llms[0].get_response = AsyncMock(
            side_effect=ProviderError("Anthropic down", provider="anthropic")
        )
        cascade.llms[1].get_response = AsyncMock(
            side_effect=ProviderError("OpenAI down", provider="openai")
        )
        cascade.llms[2].get_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_response("Test prompt")

        assert response.content == "Response from Gemini"

    async def test_raises_error_when_all_providers_fail(self, cascade):
        """Should raise ProviderError when all providers fail."""
        cascade.llms[0].get_response = AsyncMock(
            side_effect=ProviderError("Anthropic down", provider="anthropic")
        )
        cascade.llms[1].get_response = AsyncMock(
            side_effect=ProviderError("OpenAI down", provider="openai")
        )
        cascade.llms[2].get_response = AsyncMock(
            side_effect=ProviderError("Gemini down", provider="gemini")
        )

        with pytest.raises(ProviderError) as exc_info:
            await cascade.get_response("Test prompt")

        assert "All providers in cascade failed" in str(exc_info.value)
        assert exc_info.value.provider == "cascade"

    async def test_passes_arguments_to_provider(self, cascade):
        """Should pass all arguments to the provider method."""
        mock_response = MagicMock()
        cascade.llms[0].get_response = AsyncMock(return_value=mock_response)

        await cascade.get_response(
            "Test prompt",
            system_prompt="Be helpful",
            temperature=0.7,
            top_p=0.9,
        )

        cascade.llms[0].get_response.assert_called_once_with(
            user_prompt="Test prompt",
            system_prompt="Be helpful",
            temperature=0.7,
            top_p=0.9,
        )


class TestLLMCascadeGetJSONResponse:
    """Tests for LLMCascade.get_json_response method."""

    @pytest.fixture
    def cascade(self, mock_all_clients):
        """Create LLMCascade with mocked providers."""
        return LLMCascade([
            ("anthropic", "claude-sonnet-4-20250514"),
            ("openai", "gpt-4o"),
        ])

    async def test_returns_json_response_from_primary(self, cascade):
        """Should return JSON response from first provider."""
        mock_response = MagicMock()
        mock_response.content = {"key": "value"}

        cascade.llms[0].get_json_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_json_response("Return JSON")

        assert response.content == {"key": "value"}

    async def test_falls_back_on_json_response_failure(self, cascade):
        """Should fall back when first provider fails for JSON response."""
        mock_response = MagicMock()
        mock_response.content = {"fallback": "data"}

        cascade.llms[0].get_json_response = AsyncMock(
            side_effect=ProviderError("Anthropic down", provider="anthropic")
        )
        cascade.llms[1].get_json_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_json_response("Return JSON")

        assert response.content == {"fallback": "data"}


class TestLLMCascadeStructuredResponse:
    """Tests for LLMCascade.get_structured_json_response method."""

    @pytest.fixture
    def cascade(self, mock_all_clients):
        """Create LLMCascade with mocked providers."""
        return LLMCascade([
            ("anthropic", "claude-sonnet-4-20250514"),
            ("openai", "gpt-4o"),
        ])

    async def test_returns_structured_response_from_primary(self, cascade):
        """Should return structured response from first provider."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_response = MagicMock()
        mock_response.content = TestModel(name="test")

        cascade.llms[0].get_structured_json_response = AsyncMock(
            return_value=mock_response
        )

        response = await cascade.get_structured_json_response(
            response_model=TestModel,
            user_prompt="Return structured data",
        )

        assert response.content.name == "test"

    async def test_falls_back_on_structured_response_failure(self, cascade):
        """Should fall back when first provider fails for structured response."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_response = MagicMock()
        mock_response.content = TestModel(name="fallback")

        cascade.llms[0].get_structured_json_response = AsyncMock(
            side_effect=ProviderError("Anthropic down", provider="anthropic")
        )
        cascade.llms[1].get_structured_json_response = AsyncMock(
            return_value=mock_response
        )

        response = await cascade.get_structured_json_response(
            response_model=TestModel,
            user_prompt="Return structured data",
        )

        assert response.content.name == "fallback"
