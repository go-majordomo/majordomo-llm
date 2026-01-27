"""DeepSeek LLM provider implementation."""

import json
import time

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from majordomo_llm.base import (
    LLM,
    LLMJSONResponse,
    LLMResponse,
    T,
    build_schema_prompt,
    resolve_api_key,
)
from majordomo_llm.exceptions import ProviderError, ResponseParsingError


class DeepSeek(LLM):
    """DeepSeek LLM provider.

    Implements the LLM interface for DeepSeek's models using the OpenAI-compatible
    API. Supports both DeepSeek-V3 (chat) and DeepSeek-R1 (reasoner) models.

    The API key is read from the ``DEEPSEEK_API_KEY`` environment variable.

    Attributes:
        client: The async OpenAI client instance configured for DeepSeek.

    Example:
        >>> llm = DeepSeek(
        ...     model="deepseek-chat",
        ...     input_cost=0.28,
        ...     output_cost=0.42,
        ... )
        >>> response = await llm.get_response("Hello, DeepSeek!")
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        model: str,
        input_cost: float,
        output_cost: float,
        supports_temperature_top_p: bool = True,
        *,
        api_key: str | None = None,
        api_key_alias: str | None = None,
    ) -> None:
        """Initialize the DeepSeek provider.

        Args:
            model: The DeepSeek model identifier (e.g., "deepseek-chat", "deepseek-reasoner").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            api_key: Optional API key. Defaults to ``DEEPSEEK_API_KEY`` env var.
            api_key_alias: Optional human-readable name for the API key.

        Raises:
            ConfigurationError: If no API key is provided and env var is not set.
        """
        resolved_api_key = resolve_api_key(api_key, "DEEPSEEK_API_KEY", "DeepSeek")
        super().__init__(
            provider="deepseek",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            supports_temperature_top_p=supports_temperature_top_p,
            api_key=resolved_api_key,
            api_key_alias=api_key_alias,
        )
        self.client = openai.AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=self.DEEPSEEK_BASE_URL,
        )

    @retry(wait=wait_random_exponential(min=0.2, max=1), stop=stop_after_attempt(3))
    async def get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMResponse:
        """Get a plain text response from DeepSeek."""
        return await self._get_response(user_prompt, system_prompt, temperature, top_p)

    async def _get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMResponse:
        """Internal method to get a response from DeepSeek."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        start_time = time.time()
        try:
            if self.supports_temperature_top_p:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
        except openai.APIError as e:
            raise ProviderError(
                f"DeepSeek API error: {e}",
                provider="deepseek",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cached_tokens = getattr(
            getattr(response.usage, "prompt_tokens_details", None),
            "cached_tokens",
            0,
        ) or 0
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)

        return LLMResponse(
            content=response.choices[0].message.content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
        )

    async def _get_structured_response(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMJSONResponse:
        """DeepSeek-specific implementation using JSON mode for structured outputs."""
        schema = response_model.model_json_schema()
        combined_system_prompt = build_schema_prompt(schema, system_prompt)

        messages = [
            {"role": "system", "content": combined_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        start_time = time.time()
        try:
            if self.supports_temperature_top_p:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    response_format={"type": "json_object"},
                )
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
        except openai.APIError as e:
            raise ProviderError(
                f"DeepSeek API error: {e}",
                provider="deepseek",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time

        try:
            content = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            raise ResponseParsingError(
                f"Failed to parse JSON response: {e}",
                raw_content=response.choices[0].message.content,
            ) from e

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cached_tokens = getattr(
            getattr(response.usage, "prompt_tokens_details", None),
            "cached_tokens",
            0,
        ) or 0
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)

        return LLMJSONResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
        )
