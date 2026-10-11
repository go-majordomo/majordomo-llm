#!/usr/bin/env python3
"""Demo script showcasing server-side web search across providers.

This script demonstrates:
- Enabling web search via ``use_web_search=True`` on the factory
- Per-provider tool wiring (Anthropic web_search_20250305,
  OpenAI web_search_preview, Gemini google_search grounding)
- ``tool_use_cost`` accounting for Anthropic and Gemini
  (OpenAI bills web search via output tokens)
- Web search combined with structured output: the same grounded question
  answered as a validated Pydantic model

Prerequisites:
    1. Install dependencies: uv sync --all-extras
    2. Set at least one API key:
       - ANTHROPIC_API_KEY
       - OPENAI_API_KEY
       - GEMINI_API_KEY

Usage:
    uv run python examples/web_search_demo.py            # call providers directly
    uv run python examples/web_search_demo.py --gateway  # route through Steward

Gateway routing (--gateway) reads:
    - MAJORDOMO_GATEWAY_URL (defaults to http://localhost:7680)
    - MAJORDOMO_API_KEY (required)
"""

import os

from pydantic import BaseModel, Field
from shared import gateway_kwargs, run_demo, unavailable_provider_message

from majordomo_llm import get_llm_instance

# One web-search-capable model per provider. Each entry's model has
# supports_web_search: true in llm_config.yaml.
PROVIDERS: list[tuple[str, str, str]] = [
    ("anthropic", "claude-opus-5-5", "ANTHROPIC_API_KEY"),
    ("openai", "gpt-6-luna", "OPENAI_API_KEY"),
    ("gemini", "gemini-3.5-flash-lite", "GEMINI_API_KEY"),
]

# A prompt that should trigger a grounded lookup. Recency-sensitive on purpose
# so the providers actually invoke web search instead of answering from prior
# knowledge.
PROMPT = (
    "What was the most-discussed announcement at the most recent major "
    "OpenAI or Anthropic launch event? Cite the date and one source."
)


class LaunchAnnouncement(BaseModel):
    """The grounded answer, as fields instead of prose."""

    company: str = Field(description="The company that held the event")
    event: str = Field(description="Name of the launch event")
    event_date: str = Field(description="Date of the event, as YYYY-MM-DD")
    announcement: str = Field(description="The most-discussed announcement, in one sentence")
    source_url: str = Field(description="URL of one source that supports the answer")


def _print_usage(response: object) -> None:
    print(
        f"  Tokens: {response.input_tokens} in / "
        f"{response.output_tokens} out | "
        f"Time: {response.response_time:.2f}s"
    )
    print(
        f"  Cost: ${response.total_cost:.6f} "
        f"(input ${response.input_cost:.6f} + "
        f"output ${response.output_cost:.6f} + "
        f"tool ${response.tool_use_cost:.6f})"
    )


async def demo_web_search(provider: str, model: str, use_gateway: bool = False) -> bool:
    """Run a single grounded query and print the result + cost breakdown."""
    llm = get_llm_instance(
        provider,
        model,
        use_web_search=True,
        **gateway_kwargs(use_gateway, feature="web-search-demo"),
    )

    print(f"\n  [{provider}/{model}]")

    try:
        response = await llm.get_response(
            user_prompt=PROMPT,
            system_prompt="Use web search. Cite the date and a source URL.",
        )
    except Exception as e:
        print(f"  Error: {e}")
        return False

    print(f"  {response.content}")
    _print_usage(response)
    return True


async def demo_structured_web_search(
    provider: str, model: str, use_gateway: bool = False
) -> bool:
    """Ask the same grounded question and get a validated model back.

    Each provider combines the two differently: Anthropic offers the schema as a
    tool next to web search, OpenAI and Gemini attach a response schema to the
    grounded request.
    """
    llm = get_llm_instance(
        provider,
        model,
        use_web_search=True,
        **gateway_kwargs(use_gateway, feature="web-search-demo"),
    )

    print(f"\n  [{provider}/{model}]")

    try:
        response = await llm.get_structured_json_response(
            response_model=LaunchAnnouncement,
            user_prompt=PROMPT,
            system_prompt="Use web search, then answer in the requested structure.",
        )
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
        return False

    answer = response.content
    print(f"  Company      : {answer.company}")
    print(f"  Event        : {answer.event} ({answer.event_date})")
    print(f"  Announcement : {answer.announcement}")
    print(f"  Source       : {answer.source_url}")
    _print_usage(response)
    return True


async def main(use_gateway: bool = False, provider: str | None = None) -> None:
    print("=" * 80)
    print("majordomo-llm: Web Search Demo")
    print("=" * 80)

    if use_gateway:
        print("Routing through the Majordomo gateway (Steward).")

    entries = PROVIDERS if provider is None else [e for e in PROVIDERS if e[0] == provider]
    if provider is not None and not entries:
        print(unavailable_provider_message(provider, {p for p, _, _ in PROVIDERS}))
        return

    available = [(p, m) for p, m, env in entries if os.environ.get(env)]
    if not available:
        print("No API keys found. Set at least one of:")
        print("  ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY")
        return

    print(f"Available providers: {', '.join(p for p, _ in available)}")
    print(f"\nPrompt: {PROMPT}")

    failures = 0

    print("\n" + "-" * 80)
    print("Demo 1: Grounded answer as text")
    print("-" * 80)
    for provider, model in available:
        ok = await demo_web_search(provider, model, use_gateway=use_gateway)
        if not ok:
            failures += 1

    print("\n" + "-" * 80)
    print("Demo 2: Grounded answer as a structured response")
    print("-" * 80)
    for provider, model in available:
        ok = await demo_structured_web_search(provider, model, use_gateway=use_gateway)
        if not ok:
            failures += 1

    print("\n" + "=" * 80)
    total = 2 * len(available)
    print(f"Done! {total - failures}/{total} calls succeeded.")


if __name__ == "__main__":
    run_demo(main, description=__doc__, providers={p for p, _, _ in PROVIDERS})
