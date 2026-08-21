"""DeepInfra LLM provider implementation."""

from typing import ClassVar

from majordomo_llm.providers._openai_compatible import OpenAICompatibleLLM


class DeepInfra(OpenAICompatibleLLM):
    """DeepInfra LLM provider.

    Implements the LLM interface for DeepInfra-hosted open-weight models
    (DeepSeek, Kimi, GLM, Qwen, Llama) using the OpenAI-compatible chat
    completions API.

    The API key is read from the ``DEEPINFRA_API_KEY`` environment variable.

    Model IDs are HF-style ``org/Model-Name`` slugs and are passed through
    verbatim to the API. Note that DeepInfra nests its OpenAI-compatible routes
    under ``/v1/openai`` rather than ``/v1``.

    Example:
        >>> llm = DeepInfra(
        ...     model="zai-org/GLM-5.2",
        ...     input_cost=1.40,
        ...     output_cost=4.40,
        ... )
        >>> response = await llm.get_response("Hello, DeepInfra!")
    """

    PROVIDER_NAME: ClassVar[str] = "deepinfra"
    DISPLAY_NAME: ClassVar[str] = "DeepInfra"
    DEFAULT_BASE_URL: ClassVar[str] = "https://api.deepinfra.com/v1/openai"
    API_KEY_ENV: ClassVar[str] = "DEEPINFRA_API_KEY"
