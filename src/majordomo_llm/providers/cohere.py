"""Cohere LLM provider implementation."""

import json
import time

import cohere
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
    inline_schema_refs,
    resolve_api_key,
)
from majordomo_llm.exceptions import ProviderError, ResponseParsingError


class Cohere(LLM):
    """Cohere LLM provider.

    Implements the LLM interface for Cohere's models using the V2 API.
    Supports Command A, Command R+, Command R, and Command R7B models.

    The API key is read from the ``CO_API_KEY`` environment variable.

    Attributes:
        client: The async Cohere client instance.

    Example:
        >>> llm = Cohere(
        ...     model="command-a-03-2025",
        ...     input_cost=2.50,
        ...     output_cost=10.00,
        ... )
        >>> response = await llm.get_response("Hello, Cohere!")
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
        """Initialize the Cohere provider.

        Args:
            model: The Cohere model identifier (e.g., "command-a-03-2025").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            api_key: Optional API key. Defaults to ``CO_API_KEY`` env var.
            api_key_alias: Optional human-readable name for the API key.

        Raises:
            ConfigurationError: If no API key is provided and env var is not set.
        """
        resolved_api_key = resolve_api_key(api_key, "CO_API_KEY", "Cohere")
        super().__init__(
            provider="cohere",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            supports_temperature_top_p=supports_temperature_top_p,
            api_key=resolved_api_key,
            api_key_alias=api_key_alias,
        )
        self.client = cohere.AsyncClientV2(api_key=resolved_api_key)

    @retry(wait=wait_random_exponential(min=0.2, max=1), stop=stop_after_attempt(3))
    async def get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMResponse:
        """Get a plain text response from Cohere."""
        return await self._get_response(user_prompt, system_prompt, temperature, top_p)

    async def _get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMResponse:
        """Internal method to get a response from Cohere."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        start_time = time.time()
        try:
            if self.supports_temperature_top_p:
                response = await self.client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    p=top_p,
                )
            else:
                response = await self.client.chat(
                    model=self.model,
                    messages=messages,
                )
        except cohere.core.api_error.ApiError as e:
            raise ProviderError(
                f"Cohere API error: {e}",
                provider="cohere",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        input_tokens = response.usage.tokens.input_tokens
        output_tokens = response.usage.tokens.output_tokens
        cached_tokens = 0
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)

        return LLMResponse(
            content=response.message.content[0].text,
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
        """Cohere-specific implementation using JSON mode for structured outputs.

        Uses prompt-based schema injection with json_object mode since Cohere's
        json_schema validation doesn't support all JSON Schema constraints
        (e.g., minimum/maximum for numbers, enum values).

        The schema is flattened via inline_schema_refs() to remove $defs/$ref
        which Cohere's model handles poorly.
        """
        schema = response_model.model_json_schema()
        # Inline $refs to flatten the schema - Cohere struggles with $defs/$ref
        schema = inline_schema_refs(schema)
        combined_system_prompt = build_schema_prompt(schema, system_prompt)

        messages = [
            {"role": "system", "content": combined_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        start_time = time.time()
        try:
            if self.supports_temperature_top_p:
                response = await self.client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    p=top_p,
                    response_format={"type": "json_object"},
                )
            else:
                response = await self.client.chat(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
        except cohere.core.api_error.ApiError as e:
            raise ProviderError(
                f"Cohere API error: {e}",
                provider="cohere",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time

        try:
            content = json.loads(response.message.content[0].text)
        except json.JSONDecodeError as e:
            raise ResponseParsingError(
                f"Failed to parse JSON response: {e}",
                raw_content=response.message.content[0].text,
            ) from e

        input_tokens = response.usage.tokens.input_tokens
        output_tokens = response.usage.tokens.output_tokens
        cached_tokens = 0
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
