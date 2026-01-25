"""Factory functions for creating LLM instances."""

import logging
from typing import Iterator

from majordomo_llm.base import LLM
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers.anthropic import Anthropic
from majordomo_llm.providers.cohere import Cohere
from majordomo_llm.providers.deepseek import DeepSeek
from majordomo_llm.providers.gemini import Gemini
from majordomo_llm.providers.openai import OpenAI

logger = logging.getLogger(__name__)

#: Configuration mapping for all supported providers and models.
#: Costs are specified in USD per million tokens.
LLM_CONFIG: dict[str, dict] = {
    "openai": {
        "models": {
            "gpt-5": {
                "input_cost": 1.25,  # per million tokens
                "output_cost": 10.00, # per million tokens
                "supports_temperature_top_p": False
            },
            "gpt-5-mini": {
                "input_cost": 0.25,  # per million tokens
                "output_cost": 2.00,  # per million tokens
                "supports_temperature_top_p": False
            },
            "gpt-5-nano": {
                "input_cost": 0.05,  # per million tokens
                "output_cost": 0.40,  # per million tokens
                "supports_temperature_top_p": False
            },
            "gpt-4o": {
                "input_cost": 2.50,  # per million tokens
                "output_cost": 10.00  # per million tokens
            },
            "gpt-4.1": {
                "input_cost": 2.00,  # per million tokens
                "output_cost": 8.00   # per million tokens
            },
            "gpt-4.1-mini": {
                "input_cost": 0.40,  # pricing not yet available
                "output_cost": 1.60  # pricing not yet available
            },
            "gpt-4.1-nano": {
                "input_cost": 0.10,  # per million tokens
                "output_cost": 0.40  # per million tokens
            }
        },
    },
    "anthropic": {
        "models": {
            "claude-sonnet-4-5-20250929": {
                "input_cost": 3,  # per million tokens
                "output_cost": 15.00,  # per million tokens
                "supports_temperature_top_p": False
            },
            "claude-opus-4-1-20250805": {
                "input_cost": 15.00,  # per million tokens
                "output_cost": 75.00  # per million tokens
            },
            "claude-opus-4-20250514": {
                "input_cost": 15.00,  # per million tokens
                "output_cost": 75.00  # per million tokens
            },
            "claude-sonnet-4-20250514": {
                "input_cost": 3.00,   # per million tokens
                "output_cost": 15.00  # per million tokens
            },
            "claude-3-7-sonnet-latest": {
                "input_cost": 3.00,   # per million tokens
                "output_cost": 15.00  # per million tokens
            },
            "claude-3-5-haiku-latest": {
                "input_cost": 0.80,   # per million tokens (using Claude 3.5 Haiku pricing)
                "output_cost": 4.00   # per million tokens
            }
        }
    },
    "gemini": {
        "models": {
            "gemini-2.0-flash-lite": {
                "input_cost": 0.075,  # per million tokens
                "output_cost": 0.30   # per million tokens
            },
            "gemini-2.0-flash": {
                "input_cost": 0.10,   # per million tokens
                "output_cost": 0.40   # per million tokens
            },
            "gemini-2.5-flash-lite": {
                "input_cost": 0.10,  # per million tokens
                "output_cost": 0.40  # per million tokens
            },
            "gemini-2.5-flash": {
                "input_cost": 0.30,  # per million tokens
                "output_cost": 2.50  # per million tokens
            }
        }
    },
    "deepseek": {
        "models": {
            "deepseek-chat": {
                "input_cost": 0.28,   # per million tokens (cache miss)
                "output_cost": 0.42   # per million tokens
            },
            "deepseek-reasoner": {
                "input_cost": 0.28,   # per million tokens (cache miss)
                "output_cost": 1.68   # per million tokens
            }
        }
    },
    "cohere": {
        "models": {
            "command-a-03-2025": {
                "input_cost": 2.50,   # per million tokens
                "output_cost": 10.00  # per million tokens
            },
            "command-r-plus-08-2024": {
                "input_cost": 2.50,   # per million tokens
                "output_cost": 10.00  # per million tokens
            },
            "command-r-08-2024": {
                "input_cost": 0.50,   # per million tokens
                "output_cost": 1.50   # per million tokens
            },
            "command-r7b-12-2024": {
                "input_cost": 0.0375,  # per million tokens
                "output_cost": 0.15    # per million tokens
            }
        }
    }
}


def get_llm_instance(provider: str, model: str) -> LLM:
    """Create an LLM instance for the specified provider and model.

    This is the primary factory function for creating LLM instances. It handles
    provider-specific initialization and configuration lookup.

    Args:
        provider: The LLM provider name. One of: "openai", "anthropic", "gemini".
        model: The model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").

    Returns:
        An LLM instance configured for the specified provider and model.

    Raises:
        ValueError: If the provider or model is not recognized.

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
        raise ConfigurationError(f"Unknown model '{model}' for provider '{provider}'. Available: {available}")

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
