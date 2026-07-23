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
    uv run python examples/web_search_demo.py            # call providers directly
    uv run python examples/web_search_demo.py --gateway  # route through Steward

Gateway routing (--gateway) reads:
    - MAJORDOMO_GATEWAY_URL (defaults to http://localhost:7680)
    - MAJORDOMO_API_KEY (required)
"""

import os

from shared import gateway_kwargs, run_demo

from majordomo_llm import get_llm_instance

# One web-search-capable model per provider. Each entry's model has
# supports_web_search: true in llm_config.yaml.
PROVIDERS: list[tuple[str, str, str]] = [
    ("anthropic", "claude-opus-4-8-fast", "ANTHROPIC_API_KEY"),
    ("openai", "gpt-5.6-luna", "OPENAI_API_KEY"),
    ("gemini", "gemini-3.5-flash-lite", "GEMINI_API_KEY"),
]

# A prompt that should trigger a grounded lookup. Recency-sensitive on purpose
# so the providers actually invoke web search instead of answering from prior
# knowledge.
PROMPT = (
    "What was the most-discussed announcement at the most recent major "
    "OpenAI or Anthropic launch event? Cite the date and one source."
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


async def main(use_gateway: bool = False, provider: str | None = None) -> None:
    print("=" * 80)
    print("majordomo-llm: Web Search Demo")
    print("=" * 80)

    if use_gateway:
        print("Routing through the Majordomo gateway (Steward).")

    entries = PROVIDERS if provider is None else [e for e in PROVIDERS if e[0] == provider]
    if provider is not None and not entries:
        known = ", ".join(sorted({p for p, _, _ in PROVIDERS}))
        print(f"Unknown provider '{provider}'. Known providers: {known}")
        return

    available = [(p, m) for p, m, env in entries if os.environ.get(env)]
    if not available:
        print("No API keys found. Set at least one of:")
        print("  ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY")
        return

    print(f"Available providers: {', '.join(p for p, _ in available)}")
    print(f"\nPrompt: {PROMPT}")

    failures = 0
    for provider, model in available:
        ok = await demo_web_search(provider, model, use_gateway=use_gateway)
        if not ok:
            failures += 1

    print("\n" + "=" * 80)
    total = len(available)
    print(f"Done! {total - failures}/{total} succeeded.")


if __name__ == "__main__":
    run_demo(main, description=__doc__)
