"""Moonshot AI (Kimi) LLM provider implementation."""

from typing import ClassVar

from majordomo_llm.providers._openai_compatible import OpenAICompatibleLLM


class Moonshot(OpenAICompatibleLLM):
    """Moonshot AI LLM provider.

    Implements the LLM interface for Moonshot's first-party Kimi models using the
    OpenAI-compatible chat completions API. Unlike the multi-vendor inference
    platforms, Moonshot serves only its own models.

    The API key is read from the ``MOONSHOT_API_KEY`` environment variable.

    Model IDs are bare slugs (``kimi-k3``, ``kimi-k2.6``) and are passed through
    verbatim to the API.

    The default endpoint is the international platform. Mainland-China accounts
    bill through a separate platform (``https://api.moonshot.cn/v1``) in RMB at
    different rates; reach it via ``base_url``, and note that the costs
    configured here are the USD international rates.

    Example:
        >>> llm = Moonshot(
        ...     model="kimi-k3",
        ...     input_cost=3.00,
        ...     output_cost=15.00,
        ... )
        >>> response = await llm.get_response("Hello, Kimi!")
    """

    PROVIDER_NAME: ClassVar[str] = "moonshot"
    DISPLAY_NAME: ClassVar[str] = "Moonshot"
    DEFAULT_BASE_URL: ClassVar[str] = "https://api.moonshot.ai/v1"
    API_KEY_ENV: ClassVar[str] = "MOONSHOT_API_KEY"
