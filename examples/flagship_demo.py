#!/usr/bin/env python3
"""Demo script comparing two flagship models side-by-side with logging.

This script demonstrates:
- Running the same prompts through Anthropic's claude-opus-4-7 and
  OpenAI's gpt-5.5
- Logging all requests to SQLite with local file storage
- Comparing responses, costs, and performance between the two flagships

Prerequisites:
    1. Install dependencies: uv sync --all-extras
    2. Set API keys as environment variables:
       - OPENAI_API_KEY
       - ANTHROPIC_API_KEY

Usage:
    uv run python examples/flagship_demo.py            # call providers directly
    uv run python examples/flagship_demo.py --gateway  # route through Steward

Gateway routing (--gateway) reads:
    - MAJORDOMO_GATEWAY_URL (defaults to http://localhost:7680)
    - MAJORDOMO_API_KEY (required)
"""

import argparse
import asyncio
import json
import os
import traceback

from shared import EXAMPLES_DIR, clear_database, gateway_kwargs, print_summary

from majordomo_llm import get_llm_instance
from majordomo_llm.logging import FileStorageAdapter, LoggingLLM, SqliteAdapter

# Models compared in this demo. Closed-source flagships from Anthropic and
# OpenAI, plus open-weight alternatives at three reasoning effort levels so the
# cost/quality tradeoff of an open-weight migration is visible side-by-side
# against the same prompts.
FLAGSHIPS: list[tuple[str, str, str]] = [
    # Closed-source frontier flagships, one per major vendor.
    ("anthropic", "claude-opus-4-7", "ANTHROPIC_API_KEY"),
    ("openai", "gpt-5.5", "OPENAI_API_KEY"),
    ("gemini", "gemini-3.5-flash", "GEMINI_API_KEY"),
    # Native DeepSeek flagship (the reasoning model from api.deepseek.com).
    ("deepseek", "deepseek-reasoner", "DEEPSEEK_API_KEY"),
    # Anthropic Claude via Bedrock Mantle. Apples-to-apples vs the anthropic
    # entry above for Opus 4.7 — same model, different transport (AWS-native).
    ("bedrock_mantle", "anthropic.claude-opus-4-8", "AWS_BEARER_TOKEN_BEDROCK"),
    ("bedrock_mantle", "anthropic.claude-opus-4-7", "AWS_BEARER_TOKEN_BEDROCK"),
    # Open-weight alternatives via Fireworks. Same upstream model, three
    # reasoning profiles — see llm_config.yaml for the effort settings.
    (
        "fireworks",
        "accounts/fireworks/models/deepseek-v4-pro",
        "FIREWORKS_API_KEY",
    ),
    ("fireworks", "deepseek-v4-pro-reasoning", "FIREWORKS_API_KEY"),
    ("fireworks", "deepseek-v4-pro-hard", "FIREWORKS_API_KEY"),
    # Same model, same three profiles, hosted by Together — lets you compare
    # vendor-level price/latency/quality variation for an identical model.
    ("together", "deepseek-ai/DeepSeek-V4-Pro", "TOGETHER_API_KEY"),
    ("together", "deepseek-v4-pro-reasoning", "TOGETHER_API_KEY"),
    ("together", "deepseek-v4-pro-hard", "TOGETHER_API_KEY"),
]

# Output paths — separate from demo.py so the two demos do not clobber
# each other's logs.
DB_PATH = EXAMPLES_DIR / "flagship_logs.db"
STORAGE_DIR = EXAMPLES_DIR / "flagship_request_logs"


def load_prompts() -> list[dict]:
    """Load prompts from the JSON file."""
    prompts_path = EXAMPLES_DIR / "prompts.json"
    with open(prompts_path) as f:
        data = json.load(f)
    return data["prompts"]


def get_available_flagships() -> list[tuple[str, str]]:
    """Return the flagship entries whose API key env var is set."""
    available = []
    missing = []
    for provider, model, env_var in FLAGSHIPS:
        if os.environ.get(env_var):
            available.append((provider, model))
        else:
            missing.append((provider, model, env_var))

    if missing:
        print("Missing environment variables (these entries will be skipped):")
        for provider, model, env_var in missing:
            print(f"  - {provider}:{model} requires {env_var}")
        print()

    return available


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


async def main(use_gateway: bool = False) -> None:
    """Run the flagship comparison demo.

    Args:
        use_gateway: Route all requests through Majordomo Steward instead of
            calling providers directly.
    """
    print("=" * 80)
    print("majordomo-llm Demo: Flagship Comparison (claude-opus-4-7 vs gpt-5.5)")
    print("=" * 80)
    print()

    if use_gateway:
        print("Routing through the Majordomo gateway (Steward).")
        print()

    available = get_available_flagships()
    if not available:
        print(
            "No flagship API keys found. Set ANTHROPIC_API_KEY and/or "
            "OPENAI_API_KEY to run this demo."
        )
        return

    print(f"Comparing: {', '.join(f'{p}/{m}' for p, m in available)}")
    print()

    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts from prompts.json")
    for p in prompts:
        print(f"  - [{p['category']}] {p['name']}")
    print()

    # Reset previous run's logs so the summary reflects only this run.
    await clear_database(DB_PATH, STORAGE_DIR)

    print(f"Logging to: {DB_PATH}")
    print(f"Request/response bodies stored in: {STORAGE_DIR}")
    print()

    db = await SqliteAdapter.create(str(DB_PATH))
    storage = await FileStorageAdapter.create(STORAGE_DIR)
    logged_llms: list[LoggingLLM] = []

    try:
        for prompt in prompts:
            print("-" * 80)
            print(f"Prompt: {prompt['name']} ({prompt['category']})")
            print(f"User: {prompt['user'][:100]}{'...' if len(prompt['user']) > 100 else ''}")
            print("-" * 80)

            for provider, model in available:
                print(f"\n[{provider}/{model}]")

                try:
                    llm = get_llm_instance(
                        provider,
                        model,
                        **gateway_kwargs(use_gateway, feature="flagship-comparison"),
                    )
                except Exception as e:
                    print(f"  Error creating LLM: {e}")
                    continue

                logged_llm = LoggingLLM(llm, db, storage)
                logged_llms.append(logged_llm)

                result = await run_prompt(logged_llm, prompt, provider, model)

                if result and result["status"] == "success":
                    content = result["content"]
                    if len(content) > 300:
                        content = content[:300] + "..."
                    print(f"  Response: {content}")
                    print(f"  Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
                    print(f"  Cost: ${result['total_cost']:.6f} | "
                          f"Time: {result['response_time']:.2f}s")
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")

            print()

        for logged_llm in logged_llms:
            await logged_llm.flush()

        await print_summary(DB_PATH, title="FLAGSHIP COMPARISON SUMMARY")

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
    cli_args = parser.parse_args()
    asyncio.run(main(use_gateway=cli_args.gateway))
