"""Tests for the factory module."""

import pytest
from unittest.mock import patch

from majordomo_llm import (
    get_llm_instance,
    get_all_llm_instances,
    LLM_CONFIG,
)
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers.anthropic import Anthropic
from majordomo_llm.providers.openai import OpenAI
from majordomo_llm.providers.gemini import Gemini


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


class TestGetLLMInstance:
    """Tests for get_llm_instance factory function."""

    def test_creates_anthropic_provider(self, mock_all_clients):
        """Should create Anthropic instance for anthropic provider."""
        llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")

        assert isinstance(llm, Anthropic)
        assert llm.provider == "anthropic"
        assert llm.model == "claude-sonnet-4-20250514"

    def test_creates_openai_provider(self, mock_all_clients):
        """Should create OpenAI instance for openai provider."""
        llm = get_llm_instance("openai", "gpt-4o")

        assert isinstance(llm, OpenAI)
        assert llm.provider == "openai"
        assert llm.model == "gpt-4o"

    def test_creates_gemini_provider(self, mock_all_clients):
        """Should create Gemini instance for gemini provider."""
        llm = get_llm_instance("gemini", "gemini-2.5-flash")

        assert isinstance(llm, Gemini)
        assert llm.provider == "gemini"
        assert llm.model == "gemini-2.5-flash"

    def test_sets_correct_costs_from_config(self, mock_all_clients):
        """Should set input/output costs from LLM_CONFIG."""
        llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")

        expected_config = LLM_CONFIG["anthropic"]["models"]["claude-sonnet-4-20250514"]
        assert llm.input_cost == expected_config["input_cost"]
        assert llm.output_cost == expected_config["output_cost"]

    def test_sets_supports_temperature_top_p_flag(self, mock_all_clients):
        """Should set supports_temperature_top_p from config."""
        # Model with flag set to False
        llm = get_llm_instance("anthropic", "claude-sonnet-4-5-20250929")
        assert llm.supports_temperature_top_p is False

        # Model without flag (defaults to True)
        llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")
        assert llm.supports_temperature_top_p is True

    def test_raises_for_unknown_provider(self):
        """Should raise ConfigurationError for unknown provider."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_llm_instance("unknown_provider", "some-model")

        assert "Unknown LLM provider" in str(exc_info.value)
        assert "unknown_provider" in str(exc_info.value)

    def test_raises_for_unknown_model(self):
        """Should raise ConfigurationError for unknown model."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_llm_instance("anthropic", "unknown-model")

        assert "Unknown model" in str(exc_info.value)
        assert "unknown-model" in str(exc_info.value)


class TestGetAllLLMInstances:
    """Tests for get_all_llm_instances function."""

    def test_yields_instances_for_all_configured_models(self, mock_all_clients):
        """Should yield an LLM instance for each configured model."""
        instances = list(get_all_llm_instances())

        # Count expected models
        expected_count = sum(
            len(provider_config["models"])
            for provider_config in LLM_CONFIG.values()
        )

        assert len(instances) == expected_count

    def test_yields_correct_provider_types(self, mock_all_clients):
        """Should yield correct provider types."""
        instances = list(get_all_llm_instances())

        providers = {llm.provider for llm in instances}
        assert providers == {"openai", "anthropic", "gemini", "deepseek", "cohere"}


class TestLLMConfig:
    """Tests for LLM_CONFIG structure."""

    def test_all_providers_have_models(self):
        """Each provider should have at least one model configured."""
        for provider, config in LLM_CONFIG.items():
            assert "models" in config, f"{provider} missing 'models' key"
            assert len(config["models"]) > 0, f"{provider} has no models"

    def test_all_models_have_required_costs(self):
        """Each model should have input_cost and output_cost."""
        for provider, config in LLM_CONFIG.items():
            for model, model_config in config["models"].items():
                assert "input_cost" in model_config, f"{provider}/{model} missing input_cost"
                assert "output_cost" in model_config, f"{provider}/{model} missing output_cost"
                assert model_config["input_cost"] >= 0, f"{provider}/{model} invalid input_cost"
                assert model_config["output_cost"] >= 0, f"{provider}/{model} invalid output_cost"
