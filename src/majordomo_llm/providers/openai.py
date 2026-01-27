"""OpenAI LLM provider implementation."""

import time

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from majordomo_llm.base import LLM, LLMJSONResponse, LLMResponse, T, resolve_api_key
from majordomo_llm.exceptions import ProviderError


class OpenAI(LLM):
    """OpenAI LLM provider.

    Implements the LLM interface for OpenAI's GPT models, including
    support for structured outputs via the responses.parse API.

    The API key is read from the ``OPENAI_API_KEY`` environment variable.

    Attributes:
        client: The async OpenAI client instance.

    Example:
        >>> llm = OpenAI(
        ...     model="gpt-4o",
        ...     input_cost=2.5,
        ...     output_cost=10.0,
        ... )
        >>> response = await llm.get_response("Hello, GPT!")
    """

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
        """Initialize the OpenAI provider.

        Args:
            model: The GPT model identifier (e.g., "gpt-4o", "gpt-5").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            api_key: Optional API key. Defaults to ``OPENAI_API_KEY`` env var.
            api_key_alias: Optional human-readable name for the API key.

        Raises:
            ConfigurationError: If no API key is provided and env var is not set.
        """
        resolved_api_key = resolve_api_key(api_key, "OPENAI_API_KEY", "OpenAI")
        super().__init__(
            provider="openai",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            supports_temperature_top_p=supports_temperature_top_p,
            api_key=resolved_api_key,
            api_key_alias=api_key_alias,
        )
        self.client = openai.AsyncOpenAI(api_key=resolved_api_key)

    @retry(wait=wait_random_exponential(min=0.2, max=1), stop=stop_after_attempt(3))
    async def get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMResponse:
        """Get a plain text response from OpenAI."""
        return await self._get_response(user_prompt, system_prompt, temperature, top_p)

    async def _get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMResponse:
        """Internal method to get a response from OpenAI."""
        start_time = time.time()
        try:
            if self.supports_temperature_top_p:
                response = await self.client.responses.create(
                    model=self.model,
                    instructions=system_prompt,
                    input=user_prompt,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                response = await self.client.responses.create(
                    model=self.model,
                    instructions=system_prompt,
                    input=user_prompt,
                )
        except openai.APIError as e:
            raise ProviderError(
                f"OpenAI API error: {e}",
                provider="openai",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)

        return LLMResponse(
            content=response.output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=response.usage.input_tokens_details.cached_tokens,
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
        """OpenAI-specific implementation using structured outputs with JSON Schema."""
        start_time = time.time()

        try:
            if self.supports_temperature_top_p:
                response = await self.client.responses.parse(
                    model=self.model,
                    instructions=system_prompt,
                    input=user_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    text_format=response_model,
                )
            else:
                response = await self.client.responses.parse(
                    model=self.model,
                    instructions=system_prompt,
                    input=user_prompt,
                    text_format=response_model,
                )
        except openai.APIError as e:
            raise ProviderError(
                f"OpenAI API error: {e}",
                provider="openai",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)

        return LLMJSONResponse(
            content=response.output_parsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=response.usage.input_tokens_details.cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
        )
