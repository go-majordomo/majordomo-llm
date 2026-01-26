"""Cascade LLM implementation for automatic fallback between providers."""

import logging

from majordomo_llm.base import LLM, LLMJSONResponse, LLMResponse, T
from majordomo_llm.exceptions import ProviderError
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

    def __init__(self, providers: list[tuple[str, str]]) -> None:
        """Initialize the cascade with a list of providers.

        Args:
            providers: List of (provider, model) tuples in priority order.
                First provider is tried first.

        Raises:
            ValueError: If providers list is empty.
        """
        if not providers:
            raise ValueError("LLMCascade requires at least one provider")

        self.llms = [get_llm_instance(p, m) for p, m in providers]

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
    ) -> LLMResponse:
        """Get a response, falling back to next provider on failure."""
        return await self._cascade_call(
            "get_response",
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
        )

    async def get_json_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMJSONResponse:
        """Get a JSON response, falling back to next provider on failure."""
        return await self._cascade_call(
            "get_json_response",
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
        )

    async def _get_structured_response(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMJSONResponse:
        """Get a structured response, falling back to next provider on failure."""
        return await self._cascade_call(
            "get_structured_json_response",
            response_model=response_model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
        )

    async def _cascade_call(self, method_name: str, **kwargs) -> LLMResponse | LLMJSONResponse:
        """Try each provider in order until one succeeds.

        Args:
            method_name: The LLM method to call.
            **kwargs: Arguments to pass to the method.

        Returns:
            The response from the first successful provider.

        Raises:
            ProviderError: If all providers fail.
        """
        last_error: ProviderError | None = None

        for llm in self.llms:
            try:
                method = getattr(llm, method_name)
                return await method(**kwargs)
            except ProviderError as e:
                logger.warning(
                    "Provider %s/%s failed: %s. Trying next provider.",
                    llm.provider,
                    llm.model,
                    e,
                )
                last_error = e
                continue

        raise ProviderError(
            f"All providers in cascade failed. Last error: {last_error}",
            provider="cascade",
            original_error=last_error,
        )
