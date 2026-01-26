"""Factory functions for creating LLM instances."""

import importlib.resources
import logging
from collections.abc import Iterator

import yaml

from majordomo_llm.base import LLM
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers.anthropic import Anthropic
from majordomo_llm.providers.cohere import Cohere
from majordomo_llm.providers.deepseek import DeepSeek
from majordomo_llm.providers.gemini import Gemini
from majordomo_llm.providers.openai import OpenAI

logger = logging.getLogger(__name__)


def _load_llm_config() -> dict[str, dict]:
    """Load LLM configuration from the bundled YAML file."""
    config_file = importlib.resources.files("majordomo_llm").joinpath("llm_config.yaml")
    with config_file.open("r") as f:
        return yaml.safe_load(f)


#: Configuration mapping for all supported providers and models.
#: Costs are specified in USD per million tokens.
LLM_CONFIG: dict[str, dict] = _load_llm_config()


def get_llm_instance(provider: str, model: str) -> LLM:
    """Create an LLM instance for the specified provider and model.

    This is the primary factory function for creating LLM instances. It handles
    provider-specific initialization and configuration lookup.

    Args:
        provider: The LLM provider name. One of: "openai", "anthropic", "gemini",
            "deepseek", "cohere".
        model: The model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").

    Returns:
        An LLM instance configured for the specified provider and model.

    Raises:
        ConfigurationError: If the provider or model is not recognized.

    Example:
        >>> llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")
        >>> response = await llm.get_response("Hello!")
    """
    llm_config_entry = LLM_CONFIG.get(provider)
    if llm_config_entry is None:
        available = ", ".join(LLM_CONFIG.keys())
        raise ConfigurationError(f"Unknown LLM provider '{provider}'. Available: {available}")

    llm_models = llm_config_entry["models"]
    model_attributes = llm_models.get(model)
    if model_attributes is None:
        available = ", ".join(llm_models.keys())
        raise ConfigurationError(
            f"Unknown model '{model}' for provider '{provider}'. Available: {available}"
        )

    if provider == "openai":
        return OpenAI(
            model=model,
            input_cost=model_attributes["input_cost"],
            output_cost=model_attributes["output_cost"],
            supports_temperature_top_p=model_attributes.get("supports_temperature_top_p", True),
        )
    elif provider == "anthropic":
        return Anthropic(
            model=model,
            input_cost=model_attributes["input_cost"],
            output_cost=model_attributes["output_cost"],
            supports_temperature_top_p=model_attributes.get("supports_temperature_top_p", True),
        )
    elif provider == "gemini":
        return Gemini(
            model=model,
            input_cost=model_attributes["input_cost"],
            output_cost=model_attributes["output_cost"],
        )
    elif provider == "deepseek":
        return DeepSeek(
            model=model,
            input_cost=model_attributes["input_cost"],
            output_cost=model_attributes["output_cost"],
            supports_temperature_top_p=model_attributes.get("supports_temperature_top_p", True),
        )
    elif provider == "cohere":
        return Cohere(
            model=model,
            input_cost=model_attributes["input_cost"],
            output_cost=model_attributes["output_cost"],
            supports_temperature_top_p=model_attributes.get("supports_temperature_top_p", True),
        )
    else:
        raise ConfigurationError(f"Unknown LLM provider '{provider}'")


def get_all_llm_instances() -> Iterator[LLM]:
    """Create LLM instances for all configured providers and models.

    Yields LLM instances one at a time, which is useful for initialization
    or testing all available models.

    Yields:
        LLM instances for each configured provider/model combination.

    Example:
        >>> for llm in get_all_llm_instances():
        ...     print(llm.get_full_model_name())
    """
    for provider, provider_config in LLM_CONFIG.items():
        models = provider_config.get("models", {})
        for model in models:
            logger.debug("Creating LLM instance: %s/%s", provider, model)
            yield get_llm_instance(provider, model)
