"""Nebius Token Factory LLM provider implementation."""

from typing import ClassVar

from majordomo_llm.providers._openai_compatible import OpenAICompatibleLLM


class Nebius(OpenAICompatibleLLM):
    """Nebius Token Factory LLM provider.

    Implements the LLM interface for Nebius-hosted open-weight models (DeepSeek,
    Kimi, GLM, Qwen) using the OpenAI-compatible chat completions API.

    The API key is read from the ``NEBIUS_API_KEY`` environment variable.

    Model IDs are HF-style ``org/Model-Name`` slugs and are passed through
    verbatim to the API.

    Nebius renamed AI Studio to Token Factory; the default endpoint is the Token
    Factory host. Keys provisioned against the legacy AI Studio host
    (``https://api.studio.nebius.com/v1``) can reach it via ``base_url``.

    Example:
        >>> llm = Nebius(
        ...     model="moonshotai/Kimi-K3",
        ...     input_cost=3.00,
        ...     output_cost=15.00,
        ... )
        >>> response = await llm.get_response("Hello, Nebius!")
    """

    PROVIDER_NAME: ClassVar[str] = "nebius"
    DISPLAY_NAME: ClassVar[str] = "Nebius"
    DEFAULT_BASE_URL: ClassVar[str] = "https://api.tokenfactory.nebius.com/v1"
    API_KEY_ENV: ClassVar[str] = "NEBIUS_API_KEY"
