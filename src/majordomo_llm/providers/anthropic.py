"""Anthropic (Claude) LLM provider implementation."""

import logging
import time

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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from majordomo_llm.base import LLM, LLMJSONResponse, LLMResponse, T, resolve_api_key
from majordomo_llm.exceptions import ProviderError, ResponseParsingError

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
        )
        self.client = anthropic.AsyncAnthropic(api_key=resolved_api_key)

    @retry(wait=wait_random_exponential(min=0.2, max=1), stop=stop_after_attempt(3))
    async def get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMResponse:
        """Get a plain text response from Anthropic."""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant"
        start_time = time.time()

        messages = _anthropic_user_message(user_prompt)
        system_message = _anthropic_system_prompt(system_prompt)

        tools: list = []
        if self.use_web_search:
            tools.append({"type": "web_search_tool", "name": "web_search_20250305"})

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
                )
            else:
                response_message = await self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_message,
                    messages=messages,
                    tools=tools,
                    tool_choice=ToolChoiceAutoParam(type="auto"),
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

        return LLMResponse(
            content="\n".join(final_response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=response_message.usage.cache_read_input_tokens or 0,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
        )

    @retry(wait=wait_random_exponential(min=0.2, max=1), stop=stop_after_attempt(3))
    async def _get_structured_response(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
    ) -> LLMJSONResponse:
        """Anthropic-specific implementation using tool calling for structured outputs."""
        if self.model == "claude-sonnet-4-5-20250929" and self.use_web_search:
            return await self._get_structured_response_with_web_search(
                response_model=response_model,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
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
                )
            else:
                response_message = await self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    system=system_message,
                    messages=messages,
                    tools=tools,
                    tool_choice=ToolChoiceToolParam(type="tool", name="structured_response"),
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

        return LLMJSONResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=response_message.usage.cache_read_input_tokens or 0,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
        )

    async def _get_structured_response_with_web_search(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> LLMJSONResponse:
        """Get structured response with web search enabled."""
        response, execution_time = await self._structured_response_with_web_search_helper(
            response_model=response_model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
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

        return LLMJSONResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=response.usage.cache_read_input_tokens or 0,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
        )

    async def _structured_response_with_web_search_helper(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> tuple:
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
        tools = [structured_response_tool, web_search_tool]

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
            )
        except anthropic.APIError as e:
            raise ProviderError(
                f"Anthropic API error: {e}",
                provider="anthropic",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        return final_response, execution_time


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
