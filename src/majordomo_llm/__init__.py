"""Unified Python interface for multiple LLM providers with cost tracking.

majordomo-llm provides a consistent API for interacting with OpenAI, Anthropic,
and Google Gemini models, with automatic retry logic, cost calculation, and
support for structured outputs via Pydantic models.

Example:
    >>> from majordomo_llm import get_llm_instance
    >>> llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")
    >>> response = await llm.get_response("What is the capital of France?")
    >>> print(response.content)
    Paris is the capital of France.
    >>> print(f"Cost: ${response.total_cost:.6f}")
    Cost: $0.000045
"""

from majordomo_llm.base import (
    LLM,
    LLMJSONResponse,
    LLMResponse,
    LLMStructuredResponse,
    Usage,
)
from majordomo_llm.cascade import LLMCascade
from majordomo_llm.exceptions import (
    ConfigurationError,
    MajordomoError,
    ProviderError,
    ResponseParsingError,
)
from majordomo_llm.factory import (
    LLM_CONFIG,
    get_all_llm_instances,
    get_llm_instance,
)
from majordomo_llm.providers.anthropic import Anthropic
from majordomo_llm.providers.cohere import Cohere
from majordomo_llm.providers.deepseek import DeepSeek
from majordomo_llm.providers.gemini import Gemini
from majordomo_llm.providers.openai import OpenAI

__version__ = "0.1.6"

__all__ = [
    # Base classes and types
    "LLM",
    "LLMResponse",
    "LLMJSONResponse",
    "LLMStructuredResponse",
    "Usage",
    # Exceptions
    "MajordomoError",
    "ConfigurationError",
    "ProviderError",
    "ResponseParsingError",
    # Factory functions
    "get_llm_instance",
    "get_all_llm_instances",
    "LLM_CONFIG",
    # Cascade
    "LLMCascade",
    # Provider implementations
    "Anthropic",
    "Cohere",
    "DeepSeek",
    "Gemini",
    "OpenAI",
    # Version
    "__version__",
]
