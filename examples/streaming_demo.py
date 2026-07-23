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
    uv run python examples/streaming_demo.py            # call providers directly
    uv run python examples/streaming_demo.py --gateway  # route through Steward

Gateway routing (--gateway) reads:
    - MAJORDOMO_GATEWAY_URL (defaults to http://localhost:7680)
    - MAJORDOMO_API_KEY (required)
"""

import time

from shared import gateway_kwargs, get_available_providers, run_demo

from majordomo_llm import get_llm_instance

# Below this decode window (seconds), output tok/s is not meaningful: the
# provider buffered its whole response into a single late burst rather than
# streaming incrementally, so there is no decode phase to measure.
MIN_DECODE_WINDOW_S = 0.05


async def demo_streaming(
    provider: str, model: str, prompt: str, use_gateway: bool = False
) -> bool:
    """Stream a response and print chunks as they arrive.

    Returns True on success, False if the provider raised an error.
    """
    llm = get_llm_instance(
        provider, model, **gateway_kwargs(use_gateway, feature="streaming-demo")
    )

    print(f"\n  [{provider}/{model}]")
    print("  ", end="", flush=True)

    try:
        # Start the clock BEFORE get_response_stream() so TTFT includes the
        # connection + request round-trip uniformly. Some SDKs perform that
        # network I/O inside the await (OpenAI/Anthropic/Cohere/DeepSeek/
        # Fireworks/Together), others lazily during iteration (Bedrock);
        # measuring from here keeps TTFT comparable across providers.
        start = time.perf_counter()
        stream = await llm.get_response_stream(
            user_prompt=prompt,
            system_prompt="Be concise. Respond in 2-3 sentences.",
        )

        first_chunk_time = None
        chunk_count = 0

        async for chunk in stream:
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - start
            chunk_count += 1
            print(chunk, end="", flush=True)

        total = time.perf_counter() - start
        print()

        usage = stream.usage
        assert usage is not None, "usage should be finalized after iteration"
        ttft = first_chunk_time or 0.0
        # Decode throughput excludes TTFT so it reflects generation speed
        # independent of prompt length / queueing — length-agnostic output tok/s.
        # When the whole response arrives in one late burst (buffered providers
        # like Gemini), the decode window collapses and tok/s is undefined.
        decode_window = total - ttft
        if decode_window >= MIN_DECODE_WINDOW_S:
            decode_str = f"{usage.output_tokens / decode_window:.1f} tok/s"
        else:
            decode_str = "n/a (buffered)"
        print(f"  TTFT: {ttft:.2f}s | Total: {total:.2f}s | "
              f"Decode: {decode_str}")
        print(f"  Tokens: {usage.input_tokens} in / "
              f"{usage.output_tokens} out | "
              f"Cost: ${usage.total_cost:.6f}")
        if chunk_count <= 1:
            print(
                f"  Warning: only {chunk_count} chunk(s) — provider may have "
                "buffered the response instead of streaming."
            )
        return True

    except Exception as e:
        print(f"\n  Error: {e}")
        return False


async def demo_collect(
    provider: str, model: str, use_gateway: bool = False
) -> bool:
    """Demonstrate .collect() to get a full LLMResponse from a stream.

    Returns True on success, False if the provider raised an error.
    """
    llm = get_llm_instance(
        provider, model, **gateway_kwargs(use_gateway, feature="streaming-demo")
    )

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
              f"Cost: ${response.total_cost:.6f} | "
              f"Time: {response.response_time:.2f}s")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


async def main(use_gateway: bool = False, provider: str | None = None) -> None:
    """Run all streaming demos across available providers.

    Args:
        use_gateway: Route all requests through Majordomo Steward instead of
            calling providers directly.
        provider: When set, run only this provider's entries.
    """
    print("=" * 80)
    print("majordomo-llm: Streaming Demo")
    print("=" * 80)

    if use_gateway:
        print("Routing through the Majordomo gateway (Steward).")

    available_providers = get_available_providers(use_gateway=use_gateway, provider=provider)
    if not available_providers:
        print("No API keys found. Set at least one of:")
        print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY,")
        print("  DEEPSEEK_API_KEY, or CO_API_KEY")
        return

    print(f"Available providers: {', '.join(p[0] for p in available_providers)}")

    streaming_failures = 0
    collect_failures = 0

    # Demo 1: Streaming with real-time output
    print("\n" + "-" * 80)
    print("Demo 1: Streaming with real-time output")
    print("-" * 80)

    for provider, model in available_providers:
        ok = await demo_streaming(
            provider, model,
            "Explain why the sky is blue.",
            use_gateway=use_gateway,
        )
        if not ok:
            streaming_failures += 1

    # Demo 2: Collect stream into LLMResponse
    print("\n" + "-" * 80)
    print("Demo 2: Collect stream into LLMResponse")
    print("-" * 80)

    for provider, model in available_providers:
        ok = await demo_collect(provider, model, use_gateway=use_gateway)
        if not ok:
            collect_failures += 1

    print("\n" + "=" * 80)
    total = len(available_providers)
    print(
        f"Done! Streaming: {total - streaming_failures}/{total} succeeded, "
        f"Collect: {total - collect_failures}/{total} succeeded."
    )


if __name__ == "__main__":
    run_demo(main, description=__doc__)
