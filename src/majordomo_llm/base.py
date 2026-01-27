"""Base classes and types for the majordomo-llm library."""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from majordomo_llm.exceptions import ConfigurationError, ResponseParsingError


def _hash_api_key(api_key: str) -> str:
    """Compute a truncated SHA256 hash of an API key.

    Returns the first 16 characters of the hex digest, which is enough
    to identify keys without being reversible.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def resolve_api_key(api_key: str | None, env_var: str, provider_name: str) -> str:
    """Resolve an API key from parameter or environment variable.

    Args:
        api_key: Optional API key passed directly.
        env_var: Environment variable name to check if api_key is None.
        provider_name: Provider name for error message (e.g., "OpenAI", "Anthropic").

    Returns:
        The resolved API key.

    Raises:
        ConfigurationError: If no API key is found.
    """
    resolved = api_key or os.environ.get(env_var)
    if not resolved:
        raise ConfigurationError(
            f"{provider_name} API key not found. Set the {env_var} environment "
            "variable or pass api_key to the constructor."
        )
    return resolved


def inline_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline all $ref references in a JSON schema, removing $defs.

    This flattens nested model definitions so the schema is self-contained
    without JSON Schema $ref pointers, which some LLMs handle poorly.

    Args:
        schema: The JSON schema dict (from Pydantic's model_json_schema()).

    Returns:
        A new schema dict with all $ref replaced by their definitions.
    """
    import copy

    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve_refs(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj and len(obj) == 1:
                # Extract definition name from "#/$defs/EntityName"
                ref_path = obj["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path[len("#/$defs/") :]
                    if def_name in defs:
                        # Recursively resolve refs in the definition too
                        return resolve_refs(copy.deepcopy(defs[def_name]))
                return obj
            return {k: resolve_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        return obj

    return resolve_refs(schema)


def build_schema_prompt(schema: dict[str, Any], system_prompt: str | None = None) -> str:
    """Build a system prompt that includes a JSON schema instruction.

    Args:
        schema: The JSON schema dict (from Pydantic's model_json_schema()).
        system_prompt: Optional existing system prompt to prepend.

    Returns:
        Combined system prompt with schema instructions.
    """
    schema_instruction = f"""You must respond with valid JSON that matches this exact schema:
{json.dumps(schema, indent=2)}

Important: Return only the JSON object, no additional text or markdown formatting."""

    if system_prompt:
        return f"{system_prompt}\n\n{schema_instruction}"
    return schema_instruction

#: Type variable for Pydantic model types used in structured responses.
T = TypeVar("T", bound=BaseModel)

#: Number of tokens per million (used for cost calculations).
TOKENS_PER_MILLION = 1_000_000


@dataclass
class Usage:
    """Token usage and cost metrics for an LLM request.

    Attributes:
        input_tokens: Number of tokens in the input/prompt.
        output_tokens: Number of tokens in the response.
        cached_tokens: Number of tokens served from cache (provider-specific).
        input_cost: Cost for input tokens in USD.
        output_cost: Cost for output tokens in USD.
        total_cost: Total cost (input + output) in USD.
        response_time: Time taken for the request in seconds.
    """

    input_tokens: int
    output_tokens: int
    cached_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    response_time: float


@dataclass
class LLMResponse(Usage):
    """Response from an LLM containing plain text content.

    Inherits all usage metrics from :class:`Usage`.

    Attributes:
        content: The text content of the LLM response.
    """

    content: str


@dataclass
class LLMJSONResponse(Usage):
    """Response from an LLM containing parsed JSON content.

    Inherits all usage metrics from :class:`Usage`.

    Attributes:
        content: The parsed JSON content as a Python dict.
    """

    content: dict[str, Any]


@dataclass
class LLMStructuredResponse(Usage):
    """Response from an LLM containing a validated Pydantic model.

    Inherits all usage metrics from :class:`Usage`.

    Attributes:
        content: The validated Pydantic model instance.
    """

    content: BaseModel

class LLM(ABC):
    """Abstract base class for LLM provider implementations.

    Provides a unified interface for interacting with different LLM providers
    (OpenAI, Anthropic, Gemini) with automatic retry logic and cost tracking.

    Subclasses must implement the :meth:`get_response` method. Other methods
    have default implementations that can be overridden for provider-specific
    optimizations.

    Attributes:
        provider: The LLM provider name (e.g., "openai", "anthropic", "gemini").
        model: The specific model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").
        input_cost: Cost per million input tokens in USD.
        output_cost: Cost per million output tokens in USD.
        supports_temperature_top_p: Whether the model supports temperature/top_p params.
        use_web_search: Whether to enable web search (Anthropic only).
        api_key_hash: Truncated SHA256 hash of the API key (for logging).
        api_key_alias: Optional human-readable name for the API key.

    Example:
        >>> from majordomo_llm import get_llm_instance
        >>> llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")
        >>> response = await llm.get_response("What is 2+2?")
        >>> print(response.content)
        4
        >>> print(f"Cost: ${response.total_cost:.6f}")
    """

    def __init__(
        self,
        provider: str,
        model: str,
        input_cost: float,
        output_cost: float,
        supports_temperature_top_p: bool = True,
        use_web_search: bool = False,
        api_key: str | None = None,
        api_key_alias: str | None = None,
    ) -> None:
        """Initialize the LLM instance.

        Args:
            provider: The LLM provider name.
            model: The model identifier.
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            use_web_search: Enable web search capability (Anthropic only).
            api_key: The API key (used to compute hash for logging).
            api_key_alias: Optional human-readable name for the API key.
        """
        self.provider = provider
        self.model = model
        self.input_cost = input_cost
        self.output_cost = output_cost
        self.supports_temperature_top_p = supports_temperature_top_p
        self.use_web_search = use_web_search
        self.api_key_hash = _hash_api_key(api_key) if api_key else None
        self.api_key_alias = api_key_alias

    def get_full_model_name(self) -> str:
        """Get the fully qualified model name.

        Returns:
            Model name in the format "provider:model" (e.g., "anthropic:claude-sonnet-4-20250514").
        """
        return f"{self.provider}:{self.model}"

    def _calculate_costs(
        self, input_tokens: int, output_tokens: int
    ) -> tuple[float, float, float]:
        """Calculate costs for a request.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Tuple of (input_cost, output_cost, total_cost) in USD.
        """
        input_cost = (input_tokens * self.input_cost) / TOKENS_PER_MILLION
        output_cost = (output_tokens * self.output_cost) / TOKENS_PER_MILLION
        return input_cost, output_cost, input_cost + output_cost

    @abstractmethod
    async def get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMResponse:
        """Get a plain text response from the LLM.

        Args:
            user_prompt: The user's input prompt.
            system_prompt: Optional system prompt to set context/behavior.
            temperature: Sampling temperature (0.0-2.0). Lower is more deterministic.
            top_p: Nucleus sampling parameter (0.0-1.0).

        Returns:
            LLMResponse containing the text content and usage metrics.

        Raises:
            Exception: If the API request fails after retries.
        """
        raise NotImplementedError()

    @retry(wait=wait_random_exponential(min=0.2, max=1), stop=stop_after_attempt(3))
    async def get_json_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMJSONResponse:
        """Get a JSON response from the LLM.

        Automatically parses the LLM's text response as JSON.

        Args:
            user_prompt: The user's input prompt.
            system_prompt: Optional system prompt to set context/behavior.
            temperature: Sampling temperature (0.0-2.0). Lower is more deterministic.
            top_p: Nucleus sampling parameter (0.0-1.0).

        Returns:
            LLMJSONResponse containing the parsed JSON dict and usage metrics.

        Raises:
            ResponseParsingError: If the response cannot be parsed as JSON.
            Exception: If the API request fails after retries.
        """
        response = await self.get_response(user_prompt, system_prompt, temperature, top_p)
        # Strip markdown code fencing if present
        content = response.content.replace("```json", "").replace("```", "").strip()
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
            cached_tokens=response.cached_tokens,
            input_cost=response.input_cost,
            output_cost=response.output_cost,
            total_cost=response.total_cost,
            response_time=response.response_time,
        )

    async def get_structured_json_response(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMStructuredResponse:
        """Get a structured response validated against a Pydantic model.

        Uses provider-specific mechanisms (tool calling, response schemas) to
        ensure the response conforms to the specified Pydantic model schema.

        Args:
            response_model: Pydantic model class defining the expected structure.
            user_prompt: The user's input prompt.
            system_prompt: Optional system prompt to set context/behavior.
            temperature: Sampling temperature (0.0-2.0). Lower is more deterministic.
            top_p: Nucleus sampling parameter (0.0-1.0).

        Returns:
            LLMStructuredResponse containing the validated Pydantic model instance.

        Raises:
            pydantic.ValidationError: If the response doesn't match the model schema.
            Exception: If the API request fails after retries.

        Example:
            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> response = await llm.get_structured_json_response(
            ...     response_model=Person,
            ...     user_prompt="Extract: John is 30 years old",
            ... )
            >>> print(response.content.name)
            John
        """
        response = await self._get_structured_response(
            response_model=response_model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
        )
        parsed_content = response_model.model_validate(response.content)

        return LLMStructuredResponse(
            content=parsed_content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=response.cached_tokens,
            input_cost=response.input_cost,
            output_cost=response.output_cost,
            total_cost=response.total_cost,
            response_time=response.response_time,
        )

    async def _get_structured_response(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMJSONResponse:
        """Provider-specific implementation for structured responses.

        Default implementation injects the JSON schema into the system prompt.
        Providers should override this to use native structured output features.

        Args:
            response_model: Pydantic model class defining the expected structure.
            user_prompt: The user's input prompt.
            system_prompt: Optional system prompt to set context/behavior.
            temperature: Sampling temperature (0.0-2.0).
            top_p: Nucleus sampling parameter (0.0-1.0).

        Returns:
            LLMJSONResponse containing the parsed JSON content.
        """
        schema = response_model.model_json_schema()
        combined_system_prompt = build_schema_prompt(schema, system_prompt)

        if self.supports_temperature_top_p:
            return await self.get_json_response(
                user_prompt, combined_system_prompt, temperature, top_p
            )
        else:
            return await self.get_json_response(user_prompt, combined_system_prompt)
