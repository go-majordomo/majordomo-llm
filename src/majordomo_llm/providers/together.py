"""Together AI LLM provider implementation."""

import time
from collections.abc import AsyncIterator
from typing import Any

import openai

from majordomo_llm.base import (
    LLM,
    LLMResponse,
    LLMStreamResponse,
    _StreamState,
    canonicalize_json_schema_output,
    resolve_api_key,
)
from majordomo_llm.exceptions import ProviderError
from majordomo_llm.retry import retry_provider_call


class Together(LLM):
    """Together AI LLM provider.

    Implements the LLM interface for Together-hosted models (DeepSeek, Kimi,
    GLM, Qwen3, Llama 4, etc.) using the OpenAI-compatible chat completions API.

    The API key is read from the ``TOGETHER_API_KEY`` environment variable.

    Model IDs are slash-delimited HF-style (``deepseek-ai/DeepSeek-V4-Pro``) and
    are passed through verbatim to the API.

    Attributes:
        client: The async OpenAI client instance configured for Together.

    Example:
        >>> llm = Together(
        ...     model="deepseek-ai/DeepSeek-V4-Pro",
        ...     input_cost=2.10,
        ...     output_cost=4.40,
        ... )
        >>> response = await llm.get_response("Hello, Together!")
    """

    TOGETHER_BASE_URL = "https://api.together.xyz/v1"
    REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})
    THINKING_MODES = frozenset({"enabled", "disabled"})

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
        reasoning_effort: str | None = None,
        thinking: str | None = None,
    ) -> None:
        """Initialize the Together provider.

        Args:
            model: The Together model ID (e.g., "deepseek-ai/DeepSeek-V4-Pro").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            api_key: Optional API key. Defaults to ``TOGETHER_API_KEY`` env var.
            api_key_alias: Optional human-readable name for the API key.
            base_url: Optional custom base URL. Overrides TOGETHER_BASE_URL when set.
            default_headers: Optional headers sent with every request.
            reasoning_effort: Optional reasoning effort level for models that
                support it (e.g., DeepSeek V4). Forwarded to the underlying model
                via Together's OpenAI-compatible endpoint.
            thinking: Optional thinking mode ("enabled" or "disabled") for models
                that support it. Forwarded via ``extra_body``.

        Raises:
            ConfigurationError: If no API key is provided and env var is not set.
            ValueError: If reasoning_effort or thinking is invalid.
        """
        if reasoning_effort is not None and reasoning_effort not in self.REASONING_EFFORTS:
            valid = ", ".join(sorted(self.REASONING_EFFORTS))
            raise ValueError(
                f"Invalid Together reasoning_effort '{reasoning_effort}'. Valid: {valid}"
            )
        if thinking is not None and thinking not in self.THINKING_MODES:
            valid = ", ".join(sorted(self.THINKING_MODES))
            raise ValueError(f"Invalid Together thinking mode '{thinking}'. Valid: {valid}")

        resolved_api_key = resolve_api_key(api_key, "TOGETHER_API_KEY", "Together")
        super().__init__(
            provider="together",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            supports_temperature_top_p=supports_temperature_top_p,
            api_key=resolved_api_key,
            api_key_alias=api_key_alias,
            base_url=base_url,
            default_headers=default_headers,
        )
        self.client = openai.AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=self.base_url or self.TOGETHER_BASE_URL,
            default_headers=self.default_headers,
        )
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking

    def _together_request_kwargs(self) -> dict[str, Any]:
        """Build Together-specific request options for reasoning-capable models."""
        kwargs: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.thinking is not None:
            kwargs["extra_body"] = {"thinking": {"type": self.thinking}}
        return kwargs

    @retry_provider_call
    async def _get_response_impl(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Get a plain text response from Together."""
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
        """Internal method to get a response from Together."""
        messages: list[Any] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        start_time = time.time()
        request_kwargs = self._together_request_kwargs()
        try:
            if self.supports_temperature_top_p:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    extra_headers=extra_headers,
                    **request_kwargs,
                )
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    extra_headers=extra_headers,
                    **request_kwargs,
                )
        except openai.APIError as e:
            raise ProviderError(
                f"Together API error: {e}",
                provider="together",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        assert response.usage is not None
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cached_tokens = (
            getattr(
                getattr(response.usage, "prompt_tokens_details", None),
                "cached_tokens",
                0,
            )
            or 0
        )
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
            deprecation_warning=self.deprecation_warning,
        )

    async def _get_response_stream_impl(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMStreamResponse:
        """Get a streaming text response from Together."""
        messages: list[Any] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        state = _StreamState()
        request_kwargs = self._together_request_kwargs()

        try:
            if self.supports_temperature_top_p:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_headers=extra_headers,
                    **request_kwargs,
                )
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_headers=extra_headers,
                    **request_kwargs,
                )
        except openai.APIError as e:
            raise ProviderError(
                f"Together API error: {e}",
                provider="together",
                original_error=e,
            ) from e

        async def generator() -> AsyncIterator[str]:
            try:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                    if chunk.usage:
                        state.input_tokens = chunk.usage.prompt_tokens
                        state.output_tokens = chunk.usage.completion_tokens
                        state.cached_tokens = (
                            getattr(
                                getattr(chunk.usage, "prompt_tokens_details", None),
                                "cached_tokens",
                                0,
                            )
                            or 0
                        )
            except openai.APIError as e:
                raise ProviderError(
                    f"Together API error: {e}",
                    provider="together",
                    original_error=e,
                ) from e

        return LLMStreamResponse(stream=generator(), state=state, llm=self)

    # Together supports response_format={"type": "json_object"} across the board,
    # and response_format={"type": "json_schema"} on a subset of models. v1 uses
    # the json_schema shape uniformly; models that reject it surface the error as
    # ProviderError. Per-model branching is deferred until we see real failures.
    async def _get_json_schema_response(
        self,
        user_prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
        schema_name: str = "Response",
        schema_description: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Together-specific implementation using OpenAI-compatible JSON Schema."""
        messages: list[Any] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        json_schema_payload: dict[str, object] = {
            "name": schema_name,
            "schema": response_schema,
            "strict": True,
        }
        if schema_description is not None:
            json_schema_payload["description"] = schema_description

        response_format: Any = {
            "type": "json_schema",
            "json_schema": json_schema_payload,
        }

        start_time = time.time()
        request_kwargs = self._together_request_kwargs()
        try:
            if self.supports_temperature_top_p:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    response_format=response_format,
                    extra_headers=extra_headers,
                    **request_kwargs,
                )
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    extra_headers=extra_headers,
                    **request_kwargs,
                )
        except openai.APIError as e:
            raise ProviderError(
                f"Together API error: {e}",
                provider="together",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time

        assert response.usage is not None
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cached_tokens = (
            getattr(
                getattr(response.usage, "prompt_tokens_details", None),
                "cached_tokens",
                0,
            )
            or 0
        )
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)

        return LLMResponse(
            content=canonicalize_json_schema_output(
                response.choices[0].message.content or "",
                response_schema,
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
        )
