#!/usr/bin/env python3
"""Demo script showcasing server-side web search across providers.

This script demonstrates:
- Enabling web search via ``use_web_search=True`` on the factory
- Per-provider tool wiring (Anthropic web_search_20250305,
  OpenAI web_search_preview, Gemini google_search grounding)
- ``tool_use_cost`` accounting for Anthropic and Gemini
  (OpenAI bills web search via output tokens)

Prerequisites:
    1. Install dependencies: uv sync --all-extras
    2. Set at least one API key:
       - ANTHROPIC_API_KEY
       - OPENAI_API_KEY
       - GEMINI_API_KEY

Usage:
    uv run python examples/web_search_demo.py
"""

import asyncio
import os

from dotenv import load_dotenv

from majordomo_llm import get_llm_instance

load_dotenv()


# One web-search-capable model per provider. Each entry's model has
# supports_web_search: true in llm_config.yaml.
PROVIDERS: list[tuple[str, str, str]] = [
    ("anthropic", "claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY"),
    ("openai", "gpt-5.4-mini", "OPENAI_API_KEY"),
    ("gemini", "gemini-3.1-flash-lite", "GEMINI_API_KEY"),
]

# A prompt that should trigger a grounded lookup. Recency-sensitive on purpose
# so the providers actually invoke web search instead of answering from prior
# knowledge.
PROMPT = (
    "What was the most-discussed announcement at the most recent major "
    "OpenAI or Anthropic launch event? Cite the date and one source."
)


async def demo_web_search(provider: str, model: str) -> bool:
    """Run a single grounded query and print the result + cost breakdown."""
    llm = get_llm_instance(provider, model, use_web_search=True)

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
    return True


async def main() -> None:
    print("=" * 80)
    print("majordomo-llm: Web Search Demo")
    print("=" * 80)

    available = [(p, m) for p, m, env in PROVIDERS if os.environ.get(env)]
    if not available:
        print("No API keys found. Set at least one of:")
        print("  ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY")
        return

    print(f"Available providers: {', '.join(p for p, _ in available)}")
    print(f"\nPrompt: {PROMPT}")

    failures = 0
    for provider, model in available:
        ok = await demo_web_search(provider, model)
        if not ok:
            failures += 1

    print("\n" + "=" * 80)
    total = len(available)
    print(f"Done! {total - failures}/{total} succeeded.")


if __name__ == "__main__":
    asyncio.run(main())
