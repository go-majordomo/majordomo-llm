"""Google Gemini LLM provider implementation."""

import json
import os
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from majordomo_llm.base import LLM, LLMJSONResponse, LLMResponse, T
from majordomo_llm.exceptions import ConfigurationError, ProviderError, ResponseParsingError


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
        api_key: str | None = None,
    ) -> None:
        """Initialize the Gemini provider.

        Args:
            model: The Gemini model identifier (e.g., "gemini-2.5-flash").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            api_key: Optional API key. Defaults to ``GEMINI_API_KEY`` env var.

        Raises:
            ConfigurationError: If no API key is provided and env var is not set.
        """
        super().__init__(
            provider="gemini",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            supports_temperature_top_p=True,
        )
        resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ConfigurationError(
                "Gemini API key not found. Set the GEMINI_API_KEY environment "
                "variable or pass api_key to the constructor."
            )
        self.client = genai.Client(api_key=resolved_api_key)

    @retry(wait=wait_random_exponential(min=0.2, max=1), stop=stop_after_attempt(3))
    async def get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMResponse:
        """Get a plain text response from Gemini."""
        return await self._get_response(user_prompt, system_prompt, temperature, top_p)

    @retry(wait=wait_random_exponential(min=0.2, max=1), stop=stop_after_attempt(3))
    async def get_json_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMJSONResponse:
        """Get a JSON response from Gemini."""
        response = await self._get_response(user_prompt, system_prompt, temperature, top_p)
        content = response.content.replace("```json", "").replace("```", "")
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError as e:
            raise ResponseParsingError(
                f"Failed to parse JSON response: {e}",
                raw_content=response.content,
            ) from e
        return LLMJSONResponse(
            content=parsed_content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=0,
            input_cost=response.input_cost,
            output_cost=response.output_cost,
            total_cost=response.total_cost,
            response_time=response.response_time,
        )

    async def _get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMResponse:
        """Internal method to get a response from Gemini."""
        start_time = time.time()
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                ),
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
        )

    async def _get_structured_response(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMJSONResponse:
        """Gemini-specific implementation using response schema for structured outputs."""
        schema = response_model.model_json_schema()

        start_time = time.time()
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    response_schema=schema,
                    response_mime_type="application/json",
                ),
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
