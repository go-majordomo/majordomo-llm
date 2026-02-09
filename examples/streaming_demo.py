#!/usr/bin/env python3
"""Demo script showcasing streaming responses across all providers.

This script demonstrates:
- Streaming text responses with get_response_stream()
- Real-time chunk-by-chunk output
- Collecting a stream into a full LLMResponse via .collect()
- Accessing usage/cost metrics after streaming completes
- Comparing streaming performance across providers

Prerequisites:
    1. Install dependencies: uv sync --all-extras
    2. Set at least one API key:
       - OPENAI_API_KEY
       - ANTHROPIC_API_KEY
       - GEMINI_API_KEY
       - DEEPSEEK_API_KEY
       - CO_API_KEY

Usage:
    uv run python examples/streaming_demo.py
"""

import asyncio
import time

from shared import get_available_providers

from majordomo_llm import get_llm_instance


async def demo_streaming(provider: str, model: str, prompt: str) -> None:
    """Stream a response and print chunks as they arrive."""
    llm = get_llm_instance(provider, model)

    print(f"\n  [{provider}/{model}]")
    print("  ", end="", flush=True)

    try:
        stream = await llm.get_response_stream(
            user_prompt=prompt,
            system_prompt="Be concise. Respond in 2-3 sentences.",
        )

        first_chunk_time = None
        start = time.time()

        async for chunk in stream:
            if first_chunk_time is None:
                first_chunk_time = time.time() - start
            print(chunk, end="", flush=True)

        print()

        usage = stream.usage
        ttfc = first_chunk_time or 0
        print(f"  Time to first chunk: {ttfc:.2f}s | "
              f"Total: {usage.response_time:.2f}s")
        print(f"  Tokens: {usage.input_tokens} in / "
              f"{usage.output_tokens} out | "
              f"Cost: ${usage.total_cost:.6f}")

    except Exception as e:
        print(f"\n  Error: {e}")


async def demo_collect(provider: str, model: str) -> None:
    """Demonstrate .collect() to get a full LLMResponse from a stream."""
    llm = get_llm_instance(provider, model)

    print(f"\n  [{provider}/{model}]")

    try:
        stream = await llm.get_response_stream(
            user_prompt="What are the three primary colors?",
            system_prompt="Answer in one sentence.",
        )

        response = await stream.collect()

        print(f"  Content: {response.content}")
        print(f"  Tokens: {response.input_tokens} in / "
              f"{response.output_tokens} out | "
              f"Cost: ${response.total_cost:.6f}")

    except Exception as e:
        print(f"  Error: {e}")


async def main() -> None:
    """Run all streaming demos across available providers."""
    print("=" * 80)
    print("majordomo-llm: Streaming Demo")
    print("=" * 80)

    available_providers = get_available_providers()
    if not available_providers:
        print("No API keys found. Set at least one of:")
        print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY,")
        print("  DEEPSEEK_API_KEY, or CO_API_KEY")
        return

    print(f"Available providers: {', '.join(p[0] for p in available_providers)}")

    # Demo 1: Streaming with real-time output
    print("\n" + "-" * 80)
    print("Demo 1: Streaming with real-time output")
    print("-" * 80)

    for provider, model in available_providers:
        await demo_streaming(
            provider, model,
            "Explain why the sky is blue.",
        )

    # Demo 2: Collect stream into LLMResponse
    print("\n" + "-" * 80)
    print("Demo 2: Collect stream into LLMResponse")
    print("-" * 80)

    for provider, model in available_providers:
        await demo_collect(provider, model)

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
