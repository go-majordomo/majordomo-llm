#!/usr/bin/env python3
"""Demo script showcasing majordomo-llm across all providers with logging.

This script demonstrates:
- Running the same prompts across 5 different LLM providers
- Logging all requests to SQLite with local file storage
- Comparing responses, costs, and performance across providers

Prerequisites:
    1. Install dependencies: uv sync --all-extras
    2. Set API keys as environment variables:
       - OPENAI_API_KEY
       - ANTHROPIC_API_KEY
       - GEMINI_API_KEY
       - DEEPSEEK_API_KEY
       - CO_API_KEY

Usage:
    uv run python examples/demo.py            # call providers directly
    uv run python examples/demo.py --gateway  # route through Majordomo Steward

Gateway routing (--gateway) reads:
    - MAJORDOMO_GATEWAY_URL (defaults to http://localhost:7680)
    - MAJORDOMO_API_KEY (required)
"""

import argparse
import asyncio
import json
import traceback

from shared import (
    EXAMPLES_DIR,
    clear_database,
    gateway_kwargs,
    get_available_providers,
    print_summary,
)

from majordomo_llm import get_llm_instance
from majordomo_llm.logging import FileStorageAdapter, LoggingLLM, SqliteAdapter

# Output paths
DB_PATH = EXAMPLES_DIR / "llm_logs.db"
STORAGE_DIR = EXAMPLES_DIR / "request_logs"


def load_prompts() -> list[dict]:
    """Load prompts from the JSON file."""
    prompts_path = EXAMPLES_DIR / "prompts.json"
    with open(prompts_path) as f:
        data = json.load(f)
    return data["prompts"]


async def run_prompt(
    logged_llm: LoggingLLM,
    prompt: dict,
    provider: str,
    model: str,
) -> dict | None:
    """Run a single prompt and return results."""
    try:
        response = await logged_llm.get_response(
            user_prompt=prompt["user"],
            system_prompt=prompt["system"],
            temperature=0.3,
        )
        return {
            "provider": provider,
            "model": model,
            "prompt_id": prompt["id"],
            "content": response.content,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_cost": response.total_cost,
            "response_time": response.response_time,
            "status": "success",
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "provider": provider,
            "model": model,
            "prompt_id": prompt["id"],
            "content": None,
            "error": str(e),
            "status": "error",
        }


async def main(use_gateway: bool = False, provider: str | None = None) -> None:
    """Run the demo.

    Args:
        use_gateway: Route all requests through Majordomo Steward instead of
            calling providers directly.
        provider: When set, run only this provider's entries.
    """
    print("=" * 80)
    print("majordomo-llm Demo: Multi-Provider LLM with Request Logging")
    print("=" * 80)
    print()

    if use_gateway:
        print("Routing through the Majordomo gateway (Steward).")
        print()

    # Check available providers
    available_providers = get_available_providers(use_gateway=use_gateway, provider=provider)
    if not available_providers:
        print("No API keys found. Please set at least one API key to run the demo.")
        return

    print(f"Available providers: {', '.join(p[0] for p in available_providers)}")
    print()

    # Load prompts
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts from prompts.json")
    for p in prompts:
        print(f"  - [{p['category']}] {p['name']}")
    print()

    # Reset previous run's database and stored bodies so the summary
    # reflects only this run.
    await clear_database(DB_PATH, STORAGE_DIR)

    # Initialize logging
    print(f"Logging to: {DB_PATH}")
    print(f"Request/response bodies stored in: {STORAGE_DIR}")
    print()

    db = await SqliteAdapter.create(str(DB_PATH))
    storage = await FileStorageAdapter.create(STORAGE_DIR)
    logged_llms: list[LoggingLLM] = []

    try:
        # Run each prompt against each available provider
        for prompt in prompts:
            print("-" * 80)
            print(f"Prompt: {prompt['name']} ({prompt['category']})")
            print(f"User: {prompt['user'][:100]}{'...' if len(prompt['user']) > 100 else ''}")
            print("-" * 80)

            for provider, model in available_providers:
                print(f"\n[{provider}/{model}]")

                # Create LLM with logging
                try:
                    llm = get_llm_instance(
                        provider,
                        model,
                        **gateway_kwargs(use_gateway, feature="multi-provider-demo"),
                    )
                except Exception as e:
                    print(f"  Error creating LLM: {e}")
                    continue

                logged_llm = LoggingLLM(llm, db, storage)
                logged_llms.append(logged_llm)

                result = await run_prompt(logged_llm, prompt, provider, model)

                if result and result["status"] == "success":
                    content = result["content"]
                    # Truncate long responses for display
                    if len(content) > 300:
                        content = content[:300] + "..."
                    print(f"  Response: {content}")
                    print(f"  Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
                    print(f"  Cost: ${result['total_cost']:.6f} | "
                          f"Time: {result['response_time']:.2f}s")
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")

            print()

        # Flush all pending logging tasks before printing summary
        for logged_llm in logged_llms:
            await logged_llm.flush()

        # Print summary from logged data
        await print_summary(DB_PATH, title="LOGGED METRICS SUMMARY")

    finally:
        await db.close()
        await storage.close()

    print(f"\nDone! Check {STORAGE_DIR} for request/response JSON files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gateway",
        action="store_true",
        help="Route requests through Majordomo Steward "
        "(reads MAJORDOMO_GATEWAY_URL and MAJORDOMO_API_KEY).",
    )
    parser.add_argument(
        "--provider",
        help="Run only this provider's entries (e.g. anthropic, openai, gemini).",
    )
    cli_args = parser.parse_args()
    asyncio.run(main(use_gateway=cli_args.gateway, provider=cli_args.provider))
