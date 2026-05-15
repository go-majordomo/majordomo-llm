"""Cohere LLM provider implementation."""

import json
import time
from collections.abc import AsyncIterator

import cohere
from cohere import JsonObjectResponseFormatV2, SystemChatMessageV2, UserChatMessageV2
from cohere.core.request_options import RequestOptions

from majordomo_llm.base import (
    LLM,
    LLMJSONResponse,
    LLMResponse,
    LLMStreamResponse,
    T,
    _StreamState,
    build_schema_prompt,
    inline_schema_refs,
    resolve_api_key,
)
from majordomo_llm.exceptions import ProviderError, ResponseParsingError
from majordomo_llm.retry import retry_provider_call


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
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the Cohere provider.

        Args:
            model: The Cohere model identifier (e.g., "command-a-03-2025").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            api_key: Optional API key. Defaults to ``CO_API_KEY`` env var.
            api_key_alias: Optional human-readable name for the API key.
            base_url: Optional custom base URL for routing through a proxy.
            default_headers: Optional headers sent with every request.

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
            base_url=base_url,
            default_headers=default_headers,
        )
        self.client = cohere.AsyncClientV2(
            api_key=resolved_api_key,
            base_url=self.base_url,
        )

    def _cohere_request_options(
        self, extra_headers: dict[str, str] | None
    ) -> RequestOptions | None:
        """Build request_options with merged default + extra headers."""
        merged = dict(self.default_headers or {})
        if extra_headers:
            merged.update(extra_headers)
        if not merged:
            return None
        return RequestOptions(additional_headers=merged)

    @retry_provider_call
    async def get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Get a plain text response from Cohere."""
        return await self._get_response(
            user_prompt, system_prompt, temperature, top_p, extra_headers=extra_headers
        )

    async def _get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Internal method to get a response from Cohere."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        request_options = self._cohere_request_options(extra_headers)

        start_time = time.time()
        try:
            if self.supports_temperature_top_p:
                response = await self.client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    p=top_p,
                    request_options=request_options,
                )
            else:
                response = await self.client.chat(
                    model=self.model,
                    messages=messages,
                    request_options=request_options,
                )
        except cohere.core.api_error.ApiError as e:
            raise ProviderError(
                f"Cohere API error: {e}",
                provider="cohere",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        input_tokens = int(response.usage.tokens.input_tokens or 0)
        output_tokens = int(response.usage.tokens.output_tokens or 0)
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
            deprecation_warning=self.deprecation_warning,
        )

    async def get_response_stream(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMStreamResponse:
        """Get a streaming text response from Cohere."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        state = _StreamState()
        request_options = self._cohere_request_options(extra_headers)

        try:
            if self.supports_temperature_top_p:
                response = self.client.chat_stream(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    p=top_p,
                    request_options=request_options,
                )
            else:
                response = self.client.chat_stream(
                    model=self.model,
                    messages=messages,
                    request_options=request_options,
                )
        except cohere.core.api_error.ApiError as e:
            raise ProviderError(
                f"Cohere API error: {e}",
                provider="cohere",
                original_error=e,
            ) from e

        async def generator() -> AsyncIterator[str]:
            try:
                async for event in response:
                    if event.type == "content-delta":
                        yield event.delta.message.content.text
                    elif event.type == "message-end":
                        state.input_tokens = int(event.delta.usage.tokens.input_tokens or 0)
                        state.output_tokens = int(event.delta.usage.tokens.output_tokens or 0)
            except cohere.core.api_error.ApiError as e:
                raise ProviderError(
                    f"Cohere API error: {e}",
                    provider="cohere",
                    original_error=e,
                ) from e

        return LLMStreamResponse(stream=generator(), state=state, llm=self)

    @retry_provider_call
    async def _get_structured_response(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
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
            SystemChatMessageV2(content=combined_system_prompt),
            UserChatMessageV2(content=user_prompt)
        ]

        request_options = self._cohere_request_options(extra_headers)

        start_time = time.time()
        try:
            if self.supports_temperature_top_p:
                response = await self.client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    p=top_p,
                    response_format=JsonObjectResponseFormatV2(),
                    request_options=request_options,
                )
            else:
                response = await self.client.chat(
                    model=self.model,
                    messages=messages,
                    response_format=JsonObjectResponseFormatV2(),
                    request_options=request_options,
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

        input_tokens = int(response.usage.tokens.input_tokens or 0)
        output_tokens = int(response.usage.tokens.output_tokens or 0)
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
