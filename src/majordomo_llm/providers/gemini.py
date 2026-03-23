"""Google Gemini LLM provider implementation."""

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from majordomo_llm.base import (
    LLM,
    LLMJSONResponse,
    LLMResponse,
    LLMStreamResponse,
    T,
    _StreamState,
    resolve_api_key,
)
from majordomo_llm.exceptions import ProviderError, ResponseParsingError


class Gemini(LLM):
    """Google Gemini LLM provider.

    Implements the LLM interface for Google's Gemini models, including
    support for structured outputs via response schemas.

    The API key is read from the ``GEMINI_API_KEY`` environment variable.

    Attributes:
        client: The Google GenAI client instance.

    Example:
        >>> llm = Gemini(
        ...     model="gemini-2.5-flash",
        ...     input_cost=0.30,
        ...     output_cost=2.50,
        ... )
        >>> response = await llm.get_response("Hello, Gemini!")
    """

    def __init__(
        self,
        model: str,
        input_cost: float,
        output_cost: float,
        *,
        supports_temperature_top_p: bool = True,
        api_key: str | None = None,
        api_key_alias: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the Gemini provider.

        Args:
            model: The Gemini model identifier (e.g., "gemini-2.5-flash").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether the model supports temperature/top_p.
            api_key: Optional API key. Defaults to ``GEMINI_API_KEY`` env var.
            api_key_alias: Optional human-readable name for the API key.
            base_url: Optional custom base URL for routing through a proxy.
            default_headers: Optional headers sent with every request.

        Raises:
            ConfigurationError: If no API key is provided and env var is not set.
        """
        resolved_api_key = resolve_api_key(api_key, "GEMINI_API_KEY", "Gemini")
        super().__init__(
            provider="gemini",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            supports_temperature_top_p=True,
            api_key=resolved_api_key,
            api_key_alias=api_key_alias,
            base_url=base_url,
            default_headers=default_headers,
        )
        http_options = None
        if self.base_url or self.default_headers:
            http_options = types.HttpOptions(
                base_url=self.base_url,
                headers=self.default_headers,
            )
        self.client = genai.Client(api_key=resolved_api_key, http_options=http_options)

    @retry(wait=wait_random_exponential(min=0.2, max=1), stop=stop_after_attempt(3))
    async def get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Get a plain text response from Gemini."""
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
        """Internal method to get a response from Gemini."""
        start_time = time.time()
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": temperature,
            "top_p": top_p,
        }
        if extra_headers:
            config_kwargs["http_options"] = types.HttpOptions(headers=extra_headers)
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(**config_kwargs),
                contents=user_prompt,
            )
        except genai_errors.APIError as e:
            raise ProviderError(
                f"Gemini API error: {e}",
                provider="gemini",
                original_error=e,
            ) from e
        execution_time = time.time() - start_time

        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)

        return LLMResponse(
            content=response.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=0,
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
        """Get a streaming text response from Gemini."""
        state = _StreamState()
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": temperature,
            "top_p": top_p,
        }
        if extra_headers:
            config_kwargs["http_options"] = types.HttpOptions(headers=extra_headers)

        try:
            response = await self.client.aio.models.generate_content_stream(
                model=self.model,
                config=types.GenerateContentConfig(**config_kwargs),
                contents=user_prompt,
            )
        except genai_errors.APIError as e:
            raise ProviderError(
                f"Gemini API error: {e}",
                provider="gemini",
                original_error=e,
            ) from e

        async def generator() -> AsyncIterator[str]:
            try:
                async for chunk in response:
                    if chunk.text:
                        yield chunk.text
                    if chunk.usage_metadata:
                        state.input_tokens = chunk.usage_metadata.prompt_token_count or 0
                        state.output_tokens = chunk.usage_metadata.candidates_token_count or 0
            except genai_errors.APIError as e:
                raise ProviderError(
                    f"Gemini API error: {e}",
                    provider="gemini",
                    original_error=e,
                ) from e

        return LLMStreamResponse(stream=generator(), state=state, llm=self)

    async def _get_structured_response(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMJSONResponse:
        """Gemini-specific implementation using response schema for structured outputs."""
        schema = response_model.model_json_schema()

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "response_schema": schema,
            "response_mime_type": "application/json",
        }
        if extra_headers:
            config_kwargs["http_options"] = types.HttpOptions(headers=extra_headers)

        start_time = time.time()
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(**config_kwargs),
                contents=user_prompt,
            )
        except genai_errors.APIError as e:
            raise ProviderError(
                f"Gemini API error: {e}",
                provider="gemini",
                original_error=e,
            ) from e
        execution_time = time.time() - start_time

        try:
            content = json.loads(response.text)
        except json.JSONDecodeError as e:
            raise ResponseParsingError(
                f"Failed to parse JSON response: {e}",
                raw_content=response.text,
            ) from e
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)

        return LLMJSONResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=0,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
        )
