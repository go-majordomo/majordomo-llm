"""Cascade LLM implementation for automatic fallback between providers."""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

from tenacity import RetryError

from majordomo_llm.base import LLM, ImageInput, LLMJSONResponse, LLMResponse, LLMStreamResponse
from majordomo_llm.exceptions import (
    ConfigurationError,
    InputModalityUnsupported,
    MajordomoError,
    ProviderError,
    ResponseParsingError,
    ResponseTruncatedError,
)
from majordomo_llm.factory import get_llm_instance
from majordomo_llm.hooks.pipeline import HookPipeline

logger = logging.getLogger(__name__)


class CascadeMemberOptions(TypedDict, total=False):
    """Options for one :class:`LLMCascade` member, passed to :func:`get_llm_instance`.

    They override that model's ``llm_config.yaml`` values exactly as the
    same-named ``get_llm_instance`` arguments do: passing either replaces both.
    ``thinking="disabled"`` is translated to each model's own spelling of "off".
    """

    thinking: str
    reasoning_effort: str


#: One cascade member: ``(provider, model)``, or ``(provider, model, options)``.
CascadeMember = tuple[str, str] | tuple[str, str, CascadeMemberOptions]

_MEMBER_OPTION_KEYS = frozenset(CascadeMemberOptions.__annotations__)


def _member_options(member: Sequence[Any]) -> CascadeMemberOptions:
    """Validate a cascade member's shape and return its options (empty when none).

    Raises:
        ConfigurationError: If the member is not a 2- or 3-tuple, its options are
            not a mapping, or they hold a key :class:`CascadeMemberOptions` does
            not define.
    """
    if len(member) == 2:
        return {}
    if len(member) != 3:
        raise ConfigurationError(
            f"LLMCascade member {member!r} must be (provider, model) or "
            f"(provider, model, options)."
        )
    provider, model, options = member
    if not isinstance(options, Mapping):
        raise ConfigurationError(
            f"Options for cascade member '{provider}/{model}' must be a dict, "
            f"got {type(options).__name__}."
        )
    unknown = set(options) - _MEMBER_OPTION_KEYS
    if unknown:
        raise ConfigurationError(
            f"Unknown option(s) {', '.join(sorted(unknown))} for cascade member "
            f"'{provider}/{model}'. Valid: {', '.join(sorted(_MEMBER_OPTION_KEYS))}."
        )
    return cast(CascadeMemberOptions, dict(options))


class LLMCascade(LLM):
    """LLM wrapper that tries multiple providers in priority order.

    When a provider fails with a ProviderError, the next provider in the
    cascade is tried. This provides automatic failover for resilience.

    The providers list defines priority order - first provider is tried first.

    Attributes:
        llms: List of LLM instances in priority order.

    Example:
        >>> cascade = LLMCascade([
        ...     ("anthropic", "claude-sonnet-5"),  # Primary
        ...     ("openai", "gpt-6.1-sol"),                       # First fallback
        ...     ("gemini", "gemini-3.8-flash"),             # Last resort
        ... ])
        >>> response = await cascade.get_response("Hello!")

        A member can carry its own options:

        >>> cascade = LLMCascade([
        ...     ("anthropic", "claude-sonnet-5-5", {"thinking": "disabled"}),
        ...     ("together", "moonshotai/Kimi-K3", {"reasoning_effort": "low"}),
        ... ])
    """

    def __init__(
        self,
        providers: Sequence[CascadeMember],
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        hook_pipeline: HookPipeline | None = None,
    ) -> None:
        """Initialize the cascade with a list of providers.

        Args:
            providers: Members in priority order, first tried first. Each is a
                ``(provider, model)`` tuple, or ``(provider, model, options)``
                where ``options`` is a :class:`CascadeMemberOptions` dict
                (``thinking``, ``reasoning_effort``) applied to that member only.
            api_key: Optional API key. If not provided, each provider will fall
                back to its respective environment variable.
            base_url: Optional custom base URL for routing through a proxy.
            default_headers: Optional headers sent with every request.
            hook_pipeline: Optional pipeline that fires once at the cascade
                boundary. The pipeline is intentionally not propagated to
                child providers — hooks evaluate the eventual response from
                whichever provider succeeded, not each failover attempt.

        Raises:
            ValueError: If providers list is empty.
            ConfigurationError: If a member is malformed or names an unknown
                option, or if :func:`get_llm_instance` rejects a member.
        """
        if not providers:
            raise ValueError("LLMCascade requires at least one provider")

        self.llms = []
        for member in providers:
            options = _member_options(member)
            self.llms.append(
                get_llm_instance(
                    member[0],
                    member[1],
                    api_key=api_key,
                    base_url=base_url,
                    default_headers=default_headers,
                    **options,
                )
            )

        # Use primary provider's attributes for metadata
        primary = self.llms[0]
        super().__init__(
            provider="cascade",
            model=primary.model,
            input_cost=primary.input_cost,
            output_cost=primary.output_cost,
            supports_temperature_top_p=primary.supports_temperature_top_p,
            supports_image_input=any(llm.supports_image_input for llm in self.llms),
            hook_pipeline=hook_pipeline,
        )

    async def _get_response_impl(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Dispatch to the first child provider that succeeds."""
        return cast(
            LLMResponse,
            await self._cascade_call(
                "get_response",
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                extra_headers=extra_headers,
                max_tokens=max_tokens,
            ),
        )

    async def _get_response_stream_impl(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMStreamResponse:
        """Dispatch streaming to the first child provider that succeeds."""
        return cast(
            LLMStreamResponse,
            await self._cascade_call(
                "get_response_stream",
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                extra_headers=extra_headers,
                max_tokens=max_tokens,
            ),
        )

    async def _get_response_with_images_impl(
        self,
        user_prompt: str,
        images: tuple[ImageInput, ...],
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return cast(
            LLMResponse,
            await self._cascade_call(
                "get_response",
                user_prompt=user_prompt,
                images=images,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                extra_headers=extra_headers,
                max_tokens=max_tokens,
                failover_exceptions=(ProviderError, InputModalityUnsupported),
            ),
        )

    async def _get_response_stream_with_images_impl(
        self,
        user_prompt: str,
        images: tuple[ImageInput, ...],
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMStreamResponse:
        return cast(
            LLMStreamResponse,
            await self._cascade_call(
                "get_response_stream",
                user_prompt=user_prompt,
                images=images,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                extra_headers=extra_headers,
                max_tokens=max_tokens,
                failover_exceptions=(ProviderError, InputModalityUnsupported),
            ),
        )

    async def _get_json_schema_response_retried(
        self,
        user_prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
        schema_name: str = "Response",
        schema_description: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Cascade overrides the retried wrapper to skip outer retries.

        Each child provider already retries via its own ``_get_response_impl``
        / ``_get_json_schema_response_retried``. Adding another retry layer
        at the cascade boundary would multiply attempts; the cascade's job
        is failover, not retry. Failover behavior is preserved by
        ``_cascade_call``.
        """
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
                max_tokens=max_tokens,
                failover_exceptions=(ProviderError, ResponseParsingError),
            ),
        )

    async def _get_json_schema_response(
        self,
        user_prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
        schema_name: str = "Response",
        schema_description: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Unused on the cascade — ``_get_json_schema_response_retried`` dispatches directly."""
        raise NotImplementedError("LLMCascade dispatches via _get_json_schema_response_retried")

    async def _get_json_schema_response_with_images_retried(
        self,
        user_prompt: str,
        images: tuple[ImageInput, ...],
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
        schema_name: str = "Response",
        schema_description: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return cast(
            LLMResponse,
            await self._cascade_call(
                "get_json_schema_response",
                user_prompt=user_prompt,
                images=images,
                response_schema=response_schema,
                system_prompt=system_prompt,
                schema_name=schema_name,
                schema_description=schema_description,
                temperature=temperature,
                top_p=top_p,
                extra_headers=extra_headers,
                max_tokens=max_tokens,
                failover_exceptions=(
                    ProviderError,
                    ResponseParsingError,
                    InputModalityUnsupported,
                ),
            ),
        )

    async def _get_json_schema_response_with_images(
        self,
        user_prompt: str,
        images: tuple[ImageInput, ...],
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
        schema_name: str = "Response",
        schema_description: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        raise NotImplementedError(
            "LLMCascade dispatches via _get_json_schema_response_with_images_retried"
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
            The response from the first successful provider. Its ``provider`` and
            ``model`` name that member, and ``attempted`` lists every member
            tried, ending with it.

        Raises:
            ProviderError: If all providers fail.
            ResponseTruncatedError: If a member's response hits its output cap.
                Its ``usage.attempted`` includes the members that failed first.
            ResponseParsingError: If a member's response cannot be parsed on a
                path that does not fail over on it, with ``usage.attempted``
                recorded the same way. Structured calls fail over on it instead.
        """
        last_error: Exception | None = None
        failed: list[tuple[str, str]] = []

        for llm in self.llms:
            try:
                method = getattr(llm, method_name)
                result = cast(
                    LLMResponse | LLMJSONResponse | LLMStreamResponse, await method(**kwargs)
                )
            except failover_exceptions as e:
                self._log_provider_failure(llm, e)
                last_error = e
                failed.append((llm.provider, llm.model))
                continue
            except (ResponseTruncatedError, ResponseParsingError) as e:
                _prepend_attempts(e, failed)
                raise
            except RetryError as e:
                exc = e.last_attempt.exception()
                if not isinstance(exc, ProviderError):
                    if isinstance(exc, MajordomoError):
                        _prepend_attempts(exc, failed)
                    raise

                self._log_provider_failure(llm, exc)
                last_error = exc
                failed.append((llm.provider, llm.model))
                continue
            # The member records itself; the members that failed came before it.
            result.attempted = [*failed, *result.attempted]
            return result

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


def _prepend_attempts(error: MajordomoError, failed: list[tuple[str, str]]) -> None:
    """Record on a billed error's usage the members that failed before it.

    A truncation, or a parse error the cascade does not fail over on (one left
    after a member's retries ran out), ends the cascade. Its call was billed, and
    its ``usage.attempted`` should read like a response's: every member tried,
    ending with the one whose response failed.
    """
    usage = getattr(error, "usage", None)
    if usage is not None and failed:
        usage.attempted = [*failed, *usage.attempted]
