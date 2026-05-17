"""Amazon Bedrock LLM provider implementation (Converse API)."""

import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import aioboto3
from botocore.exceptions import BotoCoreError, ClientError

from majordomo_llm.base import (
    LLM,
    LLMJSONResponse,
    LLMResponse,
    LLMStreamResponse,
    T,
    _StreamState,
    canonicalize_json_schema_output,
    resolve_api_key,
)
from majordomo_llm.exceptions import ConfigurationError, ProviderError, ResponseParsingError
from majordomo_llm.retry import retry_provider_call

logger = logging.getLogger(__name__)


class Bedrock(LLM):
    """Amazon Bedrock LLM provider using the Converse API.

    Authenticates with a long-term Amazon Bedrock API key passed via the
    ``AWS_BEARER_TOKEN_BEDROCK`` environment variable (or the ``api_key``
    constructor argument). The AWS region must be supplied via ``region``
    or the ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` environment variables.

    Routes calls through ``bedrock-runtime`` ``converse`` / ``converse_stream``.

    Example:
        >>> llm = Bedrock(
        ...     model="us.anthropic.claude-sonnet-4-5-v1:0",
        ...     input_cost=3.0,
        ...     output_cost=15.0,
        ...     region="us-east-1",
        ... )
        >>> response = await llm.get_response("Hello!")
    """

    def __init__(
        self,
        model: str,
        input_cost: float,
        output_cost: float,
        supports_temperature_top_p: bool = True,
        use_web_search: bool = False,
        *,
        api_key: str | None = None,
        api_key_alias: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        region: str | None = None,
    ) -> None:
        """Initialize the Bedrock provider.

        Args:
            model: The Bedrock model identifier or inference profile ID
                (e.g., "us.anthropic.claude-sonnet-4-5-v1:0").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            use_web_search: Accepted for interface parity; Bedrock Converse has
                no native web search and this flag is ignored.
            api_key: Optional Bedrock API key. Defaults to
                ``AWS_BEARER_TOKEN_BEDROCK`` env var.
            api_key_alias: Optional human-readable name for the API key.
            base_url: Optional custom endpoint URL.
            default_headers: Accepted for interface parity; not forwarded to
                the Bedrock client.
            region: AWS region (e.g., "us-east-1"). Defaults to ``AWS_REGION``
                or ``AWS_DEFAULT_REGION`` env vars.

        Raises:
            ConfigurationError: If no API key or region can be resolved.
        """
        resolved_api_key = resolve_api_key(
            api_key, "AWS_BEARER_TOKEN_BEDROCK", "Amazon Bedrock"
        )
        resolved_region = (
            region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        )
        if not resolved_region:
            raise ConfigurationError(
                "Amazon Bedrock region not found. Pass region= to the constructor "
                "or set the AWS_REGION (or AWS_DEFAULT_REGION) environment variable."
            )

        super().__init__(
            provider="bedrock",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            supports_temperature_top_p=supports_temperature_top_p,
            use_web_search=use_web_search,
            api_key=resolved_api_key,
            api_key_alias=api_key_alias,
            base_url=base_url,
            default_headers=default_headers,
        )
        self.region = resolved_region
        # botocore reads the bearer token from this env var when signing
        # bedrock-runtime requests. Set it so explicit constructor keys work
        # even when the env var was not pre-set.
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = resolved_api_key
        self._session = aioboto3.Session(region_name=resolved_region)

    def _client(self) -> Any:
        """Open an async bedrock-runtime client context manager."""
        kwargs: dict[str, Any] = {"region_name": self.region}
        if self.base_url:
            kwargs["endpoint_url"] = self.base_url
        return self._session.client("bedrock-runtime", **kwargs)

    def _inference_config(
        self, temperature: float, top_p: float, max_tokens: int
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = {"maxTokens": max_tokens}
        if self.supports_temperature_top_p:
            cfg["temperature"] = temperature
            cfg["topP"] = top_p
        return cfg

    @retry_provider_call
    async def _get_response_impl(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Get a plain text response from Bedrock via Converse."""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant"
        if extra_headers:
            logger.debug("extra_headers ignored by Bedrock provider")

        start_time = time.time()
        try:
            async with self._client() as client:
                response = await client.converse(
                    modelId=self.model,
                    messages=_bedrock_user_message(user_prompt),
                    system=_bedrock_system_prompt(system_prompt),
                    inferenceConfig=self._inference_config(temperature, top_p, 1024),
                )
        except (ClientError, BotoCoreError) as e:
            raise ProviderError(
                f"Bedrock API error: {e}",
                provider="bedrock",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        content = _extract_text_content(response)
        input_tokens, output_tokens, cached_tokens = _extract_usage(response)
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)

        return LLMResponse(
            content=content,
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
        """Get a streaming text response from Bedrock via Converse Stream."""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant"
        if extra_headers:
            logger.debug("extra_headers ignored by Bedrock provider")

        state = _StreamState()

        async def generator() -> AsyncIterator[str]:
            try:
                async with self._client() as client:
                    response = await client.converse_stream(
                        modelId=self.model,
                        messages=_bedrock_user_message(user_prompt),
                        system=_bedrock_system_prompt(system_prompt),
                        inferenceConfig=self._inference_config(temperature, top_p, 1024),
                    )
                    async for event in response["stream"]:
                        if "contentBlockDelta" in event:
                            delta = event["contentBlockDelta"].get("delta", {})
                            text = delta.get("text")
                            if text:
                                yield text
                        elif "metadata" in event:
                            usage = event["metadata"].get("usage", {})
                            state.input_tokens = usage.get("inputTokens", state.input_tokens)
                            state.output_tokens = usage.get("outputTokens", state.output_tokens)
                            state.cached_tokens = usage.get(
                                "cacheReadInputTokens", state.cached_tokens
                            )
            except (ClientError, BotoCoreError) as e:
                raise ProviderError(
                    f"Bedrock API error: {e}",
                    provider="bedrock",
                    original_error=e,
                ) from e

        return LLMStreamResponse(stream=generator(), state=state, llm=self)

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
        """Bedrock-specific implementation using Converse tool calling."""
        if extra_headers:
            logger.debug("extra_headers ignored by Bedrock provider")

        tool_instruction = f"Use the {schema_name} tool to provide your answer."
        if system_prompt is None:
            system_prompt = f"You are a helpful assistant. {tool_instruction}"
        else:
            system_prompt = f"{system_prompt}\n\n{tool_instruction}"

        tool_config = _bedrock_tool_config(
            name=schema_name,
            description=schema_description
            or f"Provide a structured response using the {schema_name} JSON schema",
            schema=response_schema,
        )

        start_time = time.time()
        try:
            async with self._client() as client:
                response = await client.converse(
                    modelId=self.model,
                    messages=_bedrock_user_message(user_prompt),
                    system=_bedrock_system_prompt(system_prompt),
                    inferenceConfig=self._inference_config(temperature, top_p, 4096),
                    toolConfig=tool_config,
                )
        except (ClientError, BotoCoreError) as e:
            raise ProviderError(
                f"Bedrock API error: {e}",
                provider="bedrock",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        content = _extract_tool_use_input(response, schema_name)
        input_tokens, output_tokens, cached_tokens = _extract_usage(response)
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)

        return LLMResponse(
            content=canonicalize_json_schema_output(content, response_schema),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
        )

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
        """Bedrock-specific structured response via forced tool use."""
        if extra_headers:
            logger.debug("extra_headers ignored by Bedrock provider")

        tool_instruction = "Use the structured_response tool to provide your answer."
        if system_prompt is None:
            system_prompt = f"You are a helpful assistant. {tool_instruction}"
        else:
            system_prompt = f"{system_prompt}\n\n{tool_instruction}"

        schema = response_model.model_json_schema()
        tool_config = _bedrock_tool_config(
            name="structured_response",
            description=(
                f"Provide a structured response using the {response_model.__name__} format"
            ),
            schema=schema,
        )

        start_time = time.time()
        try:
            async with self._client() as client:
                response = await client.converse(
                    modelId=self.model,
                    messages=_bedrock_user_message(user_prompt),
                    system=_bedrock_system_prompt(system_prompt),
                    inferenceConfig=self._inference_config(temperature, top_p, 4096),
                    toolConfig=tool_config,
                )
        except (ClientError, BotoCoreError) as e:
            raise ProviderError(
                f"Bedrock API error: {e}",
                provider="bedrock",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        content = _extract_tool_use_input(response, "structured_response")
        input_tokens, output_tokens, cached_tokens = _extract_usage(response)
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


def _bedrock_user_message(user_prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"text": user_prompt}]}]


def _bedrock_system_prompt(system_prompt: str) -> list[dict[str, Any]]:
    return [{"text": system_prompt}]


def _bedrock_tool_config(
    name: str, description: str, schema: dict[str, Any]
) -> dict[str, Any]:
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {"json": schema},
                }
            }
        ],
        "toolChoice": {"tool": {"name": name}},
    }


def _extract_text_content(response: dict[str, Any]) -> str:
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    parts = [block["text"] for block in blocks if "text" in block]
    return "\n".join(parts)


def _extract_tool_use_input(response: dict[str, Any], tool_name: str) -> Any:
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    for block in blocks:
        tool_use = block.get("toolUse")
        if tool_use and tool_use.get("name") == tool_name:
            return tool_use.get("input")
    raise ResponseParsingError(
        f"No {tool_name} tool use found in Bedrock response",
        raw_content=str(blocks),
    )


def _extract_usage(response: dict[str, Any]) -> tuple[int, int, int]:
    usage = response.get("usage", {}) or {}
    return (
        int(usage.get("inputTokens", 0)),
        int(usage.get("outputTokens", 0)),
        int(usage.get("cacheReadInputTokens", 0) or 0),
    )
