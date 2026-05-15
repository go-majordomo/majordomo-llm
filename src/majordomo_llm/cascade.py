"""Cascade LLM implementation for automatic fallback between providers."""

import logging
from typing import Any, cast

from tenacity import RetryError

from majordomo_llm.base import LLM, LLMJSONResponse, LLMResponse, LLMStreamResponse, T
from majordomo_llm.exceptions import ProviderError, ResponseParsingError
from majordomo_llm.factory import get_llm_instance

logger = logging.getLogger(__name__)


class LLMCascade(LLM):
    """LLM wrapper that tries multiple providers in priority order.

    When a provider fails with a ProviderError, the next provider in the
    cascade is tried. This provides automatic failover for resilience.

    The providers list defines priority order - first provider is tried first.

    Attributes:
        llms: List of LLM instances in priority order.

    Example:
        >>> cascade = LLMCascade([
        ...     ("anthropic", "claude-sonnet-4-20250514"),  # Primary
        ...     ("openai", "gpt-4o"),                       # First fallback
        ...     ("gemini", "gemini-2.5-flash"),             # Last resort
        ... ])
        >>> response = await cascade.get_response("Hello!")
    """

    def __init__(
        self,
        providers: list[tuple[str, str]],
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the cascade with a list of providers.

        Args:
            providers: List of (provider, model) tuples in priority order.
                First provider is tried first.
            api_key: Optional API key. If not provided, each provider will fall
                back to its respective environment variable.
            base_url: Optional custom base URL for routing through a proxy.
            default_headers: Optional headers sent with every request.

        Raises:
            ValueError: If providers list is empty.
        """
        if not providers:
            raise ValueError("LLMCascade requires at least one provider")

        self.llms = [
            get_llm_instance(
                p,
                m,
                api_key=api_key,
                base_url=base_url,
                default_headers=default_headers,
            )
            for p, m in providers
        ]

        # Use primary provider's attributes for metadata
        primary = self.llms[0]
        super().__init__(
            provider="cascade",
            model=primary.model,
            input_cost=primary.input_cost,
            output_cost=primary.output_cost,
            supports_temperature_top_p=primary.supports_temperature_top_p,
        )

    async def get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Get a response, falling back to next provider on failure."""
        return cast(
            LLMResponse,
            await self._cascade_call(
                "get_response",
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                extra_headers=extra_headers,
            ),
        )

    async def get_json_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMJSONResponse:
        """Get a JSON response, falling back to next provider on failure."""
        return cast(
            LLMJSONResponse,
            await self._cascade_call(
                "get_json_response",
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                extra_headers=extra_headers,
            ),
        )

    async def get_response_stream(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMStreamResponse:
        """Get a streaming response, falling back to next provider on failure."""
        return cast(
            LLMStreamResponse,
            await self._cascade_call(
                "get_response_stream",
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                extra_headers=extra_headers,
            ),
        )

    async def get_json_schema_response(
        self,
        user_prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
        schema_name: str = "Response",
        schema_description: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Get a JSON-schema response, falling back to next provider on failure."""
        return cast(
            LLMResponse,
            await self._cascade_call(
                "get_json_schema_response",
                user_prompt=user_prompt,
                response_schema=response_schema,
                system_prompt=system_prompt,
                schema_name=schema_name,
                schema_description=schema_description,
                temperature=temperature,
                top_p=top_p,
                extra_headers=extra_headers,
                failover_exceptions=(ProviderError, ResponseParsingError),
                **kwargs,
            ),
        )

    async def _get_structured_response(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMJSONResponse:
        """Get a structured response, falling back to next provider on failure."""
        return cast(
            LLMJSONResponse,
            await self._cascade_call(
                "get_structured_json_response",
                response_model=response_model,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                extra_headers=extra_headers,
            ),
        )

    async def _cascade_call(
        self,
        method_name: str,
        failover_exceptions: tuple[type[Exception], ...] = (ProviderError,),
        **kwargs: Any,
    ) -> LLMResponse | LLMJSONResponse | LLMStreamResponse:
        """Try each provider in order until one succeeds.

        Args:
            method_name: The LLM method to call.
            **kwargs: Arguments to pass to the method.

        Returns:
            The response from the first successful provider.

        Raises:
            ProviderError: If all providers fail.
        """
        last_error: Exception | None = None

        for llm in self.llms:
            try:
                method = getattr(llm, method_name)
                result = await method(**kwargs)
                return cast(LLMResponse | LLMJSONResponse | LLMStreamResponse, result)
            except failover_exceptions as e:
                self._log_provider_failure(llm, e)
                last_error = e
                continue
            except RetryError as e:
                exc = e.last_attempt.exception()
                if not isinstance(exc, ProviderError):
                    raise

                self._log_provider_failure(llm, exc)
                last_error = exc
                continue

        raise ProviderError(
            f"All providers in cascade failed. Last error: {last_error}",
            provider="cascade",
            original_error=last_error,
        )

    def _log_provider_failure(self, llm: LLM, exc: Exception) -> None:
        """Log a provider failure before trying the next cascade entry."""
        logger.warning(
            "Provider %s/%s failed: %s. Trying next provider.",
            llm.provider,
            llm.model,
            exc,
        )
