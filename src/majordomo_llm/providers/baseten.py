"""Baseten Model APIs LLM provider implementation."""

from typing import ClassVar

from majordomo_llm.providers._openai_compatible import OpenAICompatibleLLM


class Baseten(OpenAICompatibleLLM):
    """Baseten Model APIs LLM provider.

    Implements the LLM interface for Baseten-hosted open-weight models (DeepSeek,
    Kimi, GLM, Inkling) using the OpenAI-compatible chat completions API.

    The API key is read from the ``BASETEN_API_KEY`` environment variable.

    Model IDs are HF-style ``org/Model-Name`` slugs and are passed through
    verbatim to the API. Dedicated (per-deployment) Baseten endpoints are reached
    by passing their URL as ``base_url``.

    Example:
        >>> llm = Baseten(
        ...     model="zai-org/GLM-5.2",
        ...     input_cost=1.40,
        ...     output_cost=4.40,
        ... )
        >>> response = await llm.get_response("Hello, Baseten!")
    """

    PROVIDER_NAME: ClassVar[str] = "baseten"
    DISPLAY_NAME: ClassVar[str] = "Baseten"
    DEFAULT_BASE_URL: ClassVar[str] = "https://inference.baseten.co/v1"
    API_KEY_ENV: ClassVar[str] = "BASETEN_API_KEY"
