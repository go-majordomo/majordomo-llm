"""Custom exceptions for majordomo-llm."""


class MajordomoError(Exception):
    """Base exception for all majordomo-llm errors.

    All custom exceptions in this library inherit from this class,
    allowing users to catch all library-specific errors with a single
    except clause.

    Example:
        >>> try:
        ...     response = await llm.get_response("Hello")
        ... except MajordomoError as e:
        ...     print(f"LLM error: {e}")
    """


class ConfigurationError(MajordomoError):
    """Raised when configuration is invalid or missing.

    This includes missing API keys, invalid provider/model combinations,
    and other configuration-related issues.

    Example:
        >>> # Missing API key
        >>> llm = get_llm_instance("openai", "gpt-4o")
        ConfigurationError: Missing OPENAI_API_KEY environment variable.
    """


class ProviderError(MajordomoError):
    """Raised when an LLM provider returns an error.

    This wraps errors from the underlying provider SDKs (OpenAI, Anthropic,
    Google) to provide a consistent interface.

    Attributes:
        provider: The provider that raised the error.
        original_error: The original exception from the provider SDK.
    """

    def __init__(self, message: str, provider: str, original_error: Exception | None = None):
        """Initialize the provider error.

        Args:
            message: Human-readable error description.
            provider: The provider name (e.g., "openai", "anthropic").
            original_error: The original exception from the provider SDK.
        """
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error


class StructuredOutputUnsupported(ProviderError):
    """Raised when a provider/model does not support structured outputs.

    Attributes:
        provider: The provider that does not support structured outputs.
        model: The model that does not support structured outputs.
    """

    def __init__(self, provider: str, model: str):
        """Initialize the unsupported structured output error.

        Args:
            provider: The provider name (e.g., "openai", "anthropic").
            model: The provider model identifier.
        """
        super().__init__(
            f"Structured outputs are not supported by provider '{provider}' model '{model}'.",
            provider=provider,
        )
        self.model = model


class ResponseParsingError(MajordomoError):
    """Raised when response parsing fails.

    This is raised when the LLM response cannot be parsed as expected,
    such as invalid JSON or missing structured output fields.

    Attributes:
        raw_content: The raw response content that failed to parse.
    """

    def __init__(self, message: str, raw_content: str | None = None):
        """Initialize the parsing error.

        Args:
            message: Human-readable error description.
            raw_content: The raw response content that failed to parse.
        """
        super().__init__(message)
        self.raw_content = raw_content


class EmptyStructuredResponseError(ResponseParsingError):
    """Raised when a structured response is schema-valid but empty.

    A forced tool call is mandatory to *invoke* but its arguments are ordinary
    generation, so a model can return ``{}`` or an object whose every field is
    ``null`` — a schema-valid non-answer. Rather than report that as a
    successful response, the library raises this so callers (and
    :class:`~majordomo_llm.cascade.LLMCascade` / the retry policy) can react. It
    is retryable: the structured-output retry wrapper re-samples before it
    surfaces.

    Subclasses :class:`ResponseParsingError`; a caller that genuinely wants to
    accept an all-null object can catch this type and read ``raw_content``.
    """
