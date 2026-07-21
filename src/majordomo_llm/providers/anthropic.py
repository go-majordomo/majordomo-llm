"""Anthropic (Claude) LLM provider implementation."""

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    TextBlockParam,
    ToolChoiceAutoParam,
    ToolChoiceToolParam,
    ToolParam,
    WebSearchTool20250305Param,
)

from majordomo_llm.base import (
    LLM,
    LLMJSONResponse,
    LLMResponse,
    LLMStreamResponse,
    T,
    _StreamState,
    canonicalize_json_schema_output,
    fill_strict_nullable_defaults,
    relax_strict_object_schema,
    resolve_api_key,
)
from majordomo_llm.exceptions import ProviderError, ResponseParsingError
from majordomo_llm.retry import retry_provider_call

logger = logging.getLogger(__name__)


class Anthropic(LLM):
    """Anthropic (Claude) LLM provider.

    Implements the LLM interface for Anthropic's Claude models, including
    support for tool calling for structured outputs and optional web search.

    The API key is read from the ``ANTHROPIC_API_KEY`` environment variable.

    Attributes:
        client: The async Anthropic client instance.

    Example:
        >>> llm = Anthropic(
        ...     model="claude-sonnet-4-20250514",
        ...     input_cost=3.0,
        ...     output_cost=15.0,
        ... )
        >>> response = await llm.get_response("Hello, Claude!")
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
    ) -> None:
        """Initialize the Anthropic provider.

        Args:
            model: The Claude model identifier (e.g., "claude-sonnet-4-20250514").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            use_web_search: Enable web search (requires claude-sonnet-4-5-20250929).
            api_key: Optional API key. Defaults to ``ANTHROPIC_API_KEY`` env var.
            api_key_alias: Optional human-readable name for the API key.
            base_url: Optional custom base URL for routing through a proxy.
            default_headers: Optional headers sent with every request.

        Raises:
            ConfigurationError: If no API key is provided and env var is not set.
        """
        resolved_api_key = resolve_api_key(api_key, "ANTHROPIC_API_KEY", "Anthropic")
        super().__init__(
            provider="anthropic",
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
        self.client = anthropic.AsyncAnthropic(
            api_key=resolved_api_key,
            base_url=self.base_url,
            default_headers=self.default_headers,
        )

    # Anthropic bills server-side web search at $10 per 1,000 requests.
    _WEB_SEARCH_COST_PER_REQUEST = 0.01

    def _compute_web_search_cost(self, response: Any) -> float:
        """Return the per-call web-search fee charged by Anthropic.

        Reads ``response.usage.server_tool_use.web_search_requests`` which is
        populated only when the web_search tool was actually invoked.
        """
        server_tool_use = getattr(response.usage, "server_tool_use", None)
        if server_tool_use is None:
            return 0.0
        requests = getattr(server_tool_use, "web_search_requests", 0) or 0
        return requests * self._WEB_SEARCH_COST_PER_REQUEST

    @retry_provider_call
    async def _get_response_impl(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Get a plain text response from Anthropic."""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant"
        start_time = time.time()

        messages = _anthropic_user_message(user_prompt)
        system_message = _anthropic_system_prompt(system_prompt)

        tools: list[Any] = []
        if self.use_web_search:
            tools.append(
                WebSearchTool20250305Param(type="web_search_20250305", name="web_search")
            )

        try:
            if self.supports_temperature_top_p:
                response_message = await self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_message,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    tools=tools,
                    tool_choice=ToolChoiceAutoParam(type="auto"),
                    extra_headers=extra_headers,
                )
            else:
                response_message = await self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_message,
                    messages=messages,
                    tools=tools,
                    tool_choice=ToolChoiceAutoParam(type="auto"),
                    extra_headers=extra_headers,
                )
        except anthropic.APIError as e:
            raise ProviderError(
                f"Anthropic API error: {e}",
                provider="anthropic",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        final_response = [c.text for c in response_message.content if c.type == "text"]

        input_tokens = response_message.usage.input_tokens
        output_tokens = response_message.usage.output_tokens
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)
        tool_use_cost = self._compute_web_search_cost(response_message)
        total_cost += tool_use_cost

        return LLMResponse(
            content="\n".join(final_response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=response_message.usage.cache_read_input_tokens or 0,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
            tool_use_cost=tool_use_cost,
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
        """Get a streaming text response from Anthropic."""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant"

        state = _StreamState()
        messages = _anthropic_user_message(user_prompt)
        system_message = _anthropic_system_prompt(system_prompt)

        try:
            if self.supports_temperature_top_p:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_message,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                    extra_headers=extra_headers,
                )
            else:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_message,
                    messages=messages,
                    stream=True,
                    extra_headers=extra_headers,
                )
        except anthropic.APIError as e:
            raise ProviderError(
                f"Anthropic API error: {e}",
                provider="anthropic",
                original_error=e,
            ) from e

        async def generator() -> AsyncIterator[str]:
            try:
                async for event in response:
                    if event.type == "message_start":
                        state.input_tokens = event.message.usage.input_tokens
                        state.cached_tokens = event.message.usage.cache_read_input_tokens or 0
                    elif event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield event.delta.text
                    elif event.type == "message_delta":
                        state.output_tokens = event.usage.output_tokens
            except anthropic.APIError as e:
                raise ProviderError(
                    f"Anthropic API error: {e}",
                    provider="anthropic",
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
        """Anthropic-specific implementation using forced tool calling."""
        if self.use_web_search:
            response, execution_time = await self._json_schema_response_with_web_search_helper(
                user_prompt=user_prompt,
                response_schema=response_schema,
                system_prompt=system_prompt,
                schema_name=schema_name,
                schema_description=schema_description,
                extra_headers=extra_headers,
            )
        else:
            tool_instruction = f"Use the {schema_name} tool to provide your answer."
            if system_prompt is None:
                system_prompt = f"You are a helpful assistant. {tool_instruction}"
            else:
                system_prompt = f"{system_prompt}\n\n{tool_instruction}"

            messages = _anthropic_user_message(user_prompt)
            system_message = _anthropic_system_prompt(system_prompt)
            tools = [
                ToolParam(
                    name=schema_name,
                    description=schema_description
                    or f"Provide a structured response using the {schema_name} JSON schema",
                    input_schema=relax_strict_object_schema(response_schema),
                )
            ]

            start_time = time.time()
            try:
                if self.supports_temperature_top_p:
                    response = await self.client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        system=system_message,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        tools=tools,
                        tool_choice=ToolChoiceToolParam(type="tool", name=schema_name),
                        extra_headers=extra_headers,
                    )
                else:
                    response = await self.client.messages.create(
                        model=self.model,
                        max_tokens=8192,
                        system=system_message,
                        messages=messages,
                        tools=tools,
                        tool_choice=ToolChoiceToolParam(type="tool", name=schema_name),
                        extra_headers=extra_headers,
                    )
            except anthropic.APIError as e:
                raise ProviderError(
                    f"Anthropic API error: {e}",
                    provider="anthropic",
                    original_error=e,
                ) from e

            execution_time = time.time() - start_time

        content = _extract_tool_use_content(response.content, schema_name)
        content = fill_strict_nullable_defaults(content, response_schema)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)
        tool_use_cost = self._compute_web_search_cost(response)
        total_cost += tool_use_cost

        return LLMResponse(
            content=canonicalize_json_schema_output(content, response_schema),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=response.usage.cache_read_input_tokens or 0,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
            tool_use_cost=tool_use_cost,
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
        """Anthropic-specific implementation using tool calling for structured outputs."""
        if self.use_web_search:
            return await self._get_structured_response_with_web_search(
                response_model=response_model,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                extra_headers=extra_headers,
            )

        schema = response_model.model_json_schema()

        tool_instruction = "Use the structured_response tool to provide your answer."
        if system_prompt is None:
            system_prompt = f"You are a helpful assistant. {tool_instruction}"
        else:
            system_prompt = f"{system_prompt}\n\n{tool_instruction}"

        messages = _anthropic_user_message(user_prompt)
        system_message = _anthropic_system_prompt(system_prompt)
        tool_desc = f"Provide a structured response using the {response_model.__name__} format"
        tools = [
            ToolParam(
                name="structured_response",
                description=tool_desc,
                input_schema=schema,
            )
        ]

        start_time = time.time()
        try:
            if self.supports_temperature_top_p:
                response_message = await self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_message,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    tools=tools,
                    tool_choice=ToolChoiceToolParam(type="tool", name="structured_response"),
                    extra_headers=extra_headers,
                )
            else:
                response_message = await self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    system=system_message,
                    messages=messages,
                    tools=tools,
                    tool_choice=ToolChoiceToolParam(type="tool", name="structured_response"),
                    extra_headers=extra_headers,
                )
        except anthropic.APIError as e:
            raise ProviderError(
                f"Anthropic API error: {e}",
                provider="anthropic",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time

        # Extract the tool use content
        content = None
        for block in response_message.content:
            if block.type == "tool_use" and block.name == "structured_response":
                content = block.input
                break

        if content is None:
            raise ResponseParsingError(
                "No structured response tool use found in Anthropic response",
                raw_content=str(response_message.content),
            )

        input_tokens = response_message.usage.input_tokens
        output_tokens = response_message.usage.output_tokens
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)
        tool_use_cost = self._compute_web_search_cost(response_message)
        total_cost += tool_use_cost

        return LLMJSONResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=response_message.usage.cache_read_input_tokens or 0,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
            tool_use_cost=tool_use_cost,
        )

    async def _get_structured_response_with_web_search(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMJSONResponse:
        """Get structured response with web search enabled."""
        response, execution_time = await self._structured_response_with_web_search_helper(
            response_model=response_model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            extra_headers=extra_headers,
        )

        content = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "structured_response":
                content = block.input
                break

        if content is None:
            raise ResponseParsingError(
                "No structured response tool use found in Anthropic response",
                raw_content=str(response.content),
            )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)
        tool_use_cost = self._compute_web_search_cost(response)
        total_cost += tool_use_cost

        return LLMJSONResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=response.usage.cache_read_input_tokens or 0,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
            tool_use_cost=tool_use_cost,
        )

    async def _structured_response_with_web_search_helper(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[Any, float]:
        """Helper for web search with structured response."""
        schema = response_model.model_json_schema()
        structured_response_tool = ToolParam(
            name="structured_response",
            description=f"Provide a structured response using the {response_model.__name__} format",
            input_schema=schema,
        )
        web_search_tool = WebSearchTool20250305Param(
            name="web_search",
            type="web_search_20250305",
        )
        tools: list[Any] = [structured_response_tool, web_search_tool]

        tool_instruction = "Use the structured_response tool to provide your answer."
        if system_prompt is None:
            system_prompt = f"You are a helpful assistant. {tool_instruction}"
        else:
            system_prompt = f"{system_prompt}\n\n{tool_instruction}"

        messages = _anthropic_user_message(user_prompt)
        system_message = _anthropic_system_prompt(system_prompt)

        start_time = time.time()
        current_messages = messages.copy()
        search_count = 0

        try:
            while search_count < 3:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    system=system_message,
                    messages=current_messages,
                    tools=tools,
                    tool_choice=ToolChoiceAutoParam(type="auto"),
                    extra_headers=extra_headers,
                )

                # Check what tool was used
                if response.stop_reason == "tool_use":
                    tool_uses = [b for b in response.content if b.type == "tool_use"]

                    # If structured_response was used, we're done!
                    if any(t.name == "structured_response" for t in tool_uses):
                        execution_time = time.time() - start_time
                        return response, execution_time

                    # If web_search was used, continue conversation
                    if any(t.name == "web_search" for t in tool_uses):
                        logger.info("Web search initiated (turn %d)", search_count + 1)
                        search_count += 1

                        # Add assistant response
                        current_messages.append({
                            "role": "assistant",
                            "content": response.content,
                        })

                        # Add continuation prompt
                        current_messages.append({
                            "role": "user",
                            "content": (
                            "Continue with your analysis. Use the structured_response "
                            "tool when ready to generate the final output."
                        ),
                        })
                        continue
                break

            final_response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=_anthropic_system_prompt(system_prompt),
                messages=current_messages,
                tools=[structured_response_tool],
                tool_choice=ToolChoiceToolParam(type="tool", name="structured_response"),
                extra_headers=extra_headers,
            )
        except anthropic.APIError as e:
            raise ProviderError(
                f"Anthropic API error: {e}",
                provider="anthropic",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        return final_response, execution_time

    async def _json_schema_response_with_web_search_helper(
        self,
        user_prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
        schema_name: str = "Response",
        schema_description: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[Any, float]:
        """Helper for web search with raw JSON-schema structured response."""
        structured_response_tool = ToolParam(
            name=schema_name,
            description=schema_description
            or f"Provide a structured response using the {schema_name} JSON schema",
            input_schema=relax_strict_object_schema(response_schema),
        )
        web_search_tool = WebSearchTool20250305Param(
            name="web_search",
            type="web_search_20250305",
        )
        tools: list[Any] = [structured_response_tool, web_search_tool]

        tool_instruction = f"Use the {schema_name} tool to provide your answer."
        if system_prompt is None:
            system_prompt = f"You are a helpful assistant. {tool_instruction}"
        else:
            system_prompt = f"{system_prompt}\n\n{tool_instruction}"

        messages = _anthropic_user_message(user_prompt)
        system_message = _anthropic_system_prompt(system_prompt)

        start_time = time.time()
        current_messages = messages.copy()
        search_count = 0

        try:
            while search_count < 3:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    system=system_message,
                    messages=current_messages,
                    tools=tools,
                    tool_choice=ToolChoiceAutoParam(type="auto"),
                    extra_headers=extra_headers,
                )

                if response.stop_reason == "tool_use":
                    tool_uses = [block for block in response.content if block.type == "tool_use"]
                    if any(tool_use.name == schema_name for tool_use in tool_uses):
                        execution_time = time.time() - start_time
                        return response, execution_time

                    if any(tool_use.name == "web_search" for tool_use in tool_uses):
                        logger.info("Web search initiated (turn %d)", search_count + 1)
                        search_count += 1
                        current_messages.append({"role": "assistant", "content": response.content})
                        current_messages.append({
                            "role": "user",
                            "content": (
                                "Continue with your analysis. Use the structured response "
                                "tool when ready to generate the final output."
                            ),
                        })
                        continue
                break

            final_response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=_anthropic_system_prompt(system_prompt),
                messages=current_messages,
                tools=[structured_response_tool],
                tool_choice=ToolChoiceToolParam(type="tool", name=schema_name),
                extra_headers=extra_headers,
            )
        except anthropic.APIError as e:
            raise ProviderError(
                f"Anthropic API error: {e}",
                provider="anthropic",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        return final_response, execution_time


def _extract_tool_use_content(content_blocks: list[Any], tool_name: str) -> Any:
    """Extract a named Anthropic tool_use input from response content blocks."""
    for block in content_blocks:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input
    raise ResponseParsingError(
        f"No {tool_name} tool use found in Anthropic response",
        raw_content=str(content_blocks),
    )


def _anthropic_system_prompt(system_prompt: str) -> list[TextBlockParam]:
    """Create Anthropic system prompt with cache control."""
    return [
        TextBlockParam(
            type="text",
            text=system_prompt,
            cache_control=CacheControlEphemeralParam(type="ephemeral"),
        )
    ]


def _anthropic_user_message(user_prompt: str) -> list[MessageParam]:
    """Create Anthropic user message."""
    return [
        MessageParam(
            role="user",
            content=user_prompt,
        )
    ]
