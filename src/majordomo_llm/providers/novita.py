"""Novita AI LLM provider implementation."""

from typing import ClassVar

from majordomo_llm.providers._openai_compatible import OpenAICompatibleLLM


class Novita(OpenAICompatibleLLM):
    """Novita AI LLM provider.

    Implements the LLM interface for Novita-hosted open-weight models (DeepSeek,
    Kimi, GLM, Qwen, Llama) using the OpenAI-compatible chat completions API.

    The API key is read from the ``NOVITA_API_KEY`` environment variable.

    Model IDs are lowercase slash-delimited slugs and are passed through verbatim
    to the API. Note the organization prefix differs from the HF-style platforms:
    Novita uses ``deepseek/deepseek-v4-pro`` where Baseten, Nebius, and DeepInfra
    use ``deepseek-ai/DeepSeek-V4-Pro``.

    Novita also serves the same routes under ``/openai`` and ``/v3/openai``; the
    default below is the one its API reference documents.

    Example:
        >>> llm = Novita(
        ...     model="zai-org/glm-5.2",
        ...     input_cost=1.40,
        ...     output_cost=4.40,
        ... )
        >>> response = await llm.get_response("Hello, Novita!")
    """

    PROVIDER_NAME: ClassVar[str] = "novita"
    DISPLAY_NAME: ClassVar[str] = "Novita"
    DEFAULT_BASE_URL: ClassVar[str] = "https://api.novita.ai/openai/v1"
    API_KEY_ENV: ClassVar[str] = "NOVITA_API_KEY"
