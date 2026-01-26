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
    uv run python examples/demo.py
"""

import asyncio
import json
import os
import traceback
from pathlib import Path

import aiosqlite
from dotenv import load_dotenv

from majordomo_llm import get_llm_instance
from majordomo_llm.logging import FileStorageAdapter, LoggingLLM, SqliteAdapter

# Load API keys from .env file
load_dotenv()

# One model per provider
PROVIDERS = [
    ("openai", "gpt-4.1-mini"),
    ("anthropic", "claude-3-5-haiku-latest"),
    ("gemini", "gemini-2.0-flash"),
    ("deepseek", "deepseek-chat"),
    ("cohere", "command-r-08-2024"),
]

# Output paths
EXAMPLES_DIR = Path(__file__).parent
DB_PATH = EXAMPLES_DIR / "llm_logs.db"
STORAGE_DIR = EXAMPLES_DIR / "request_logs"


def load_prompts() -> list[dict]:
    """Load prompts from the JSON file."""
    prompts_path = EXAMPLES_DIR / "prompts.json"
    with open(prompts_path) as f:
        data = json.load(f)
    return data["prompts"]


def check_api_keys() -> list[tuple[str, str]]:
    """Check which providers have API keys configured."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "cohere": "CO_API_KEY",
    }
    available = []
    missing = []
    for provider, model in PROVIDERS:
        env_var = key_map[provider]
        if os.environ.get(env_var):
            available.append((provider, model))
        else:
            missing.append((provider, env_var))

    if missing:
        print("Missing API keys (these providers will be skipped):")
        for provider, env_var in missing:
            print(f"  - {provider}: set {env_var}")
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


async def print_summary(db_path: Path) -> None:
    """Query SQLite and print a summary of all requests."""
    print("\n" + "=" * 80)
    print("LOGGED METRICS SUMMARY")
    print("=" * 80)

    async with aiosqlite.connect(db_path) as db:
        # Summary by provider
        cursor = await db.execute("""
            SELECT
                provider,
                model,
                COUNT(*) as requests,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(total_cost) as total_cost,
                AVG(response_time) as avg_response_time
            FROM llm_requests
            WHERE status = 'success'
            GROUP BY provider, model
            ORDER BY provider
        """)
        rows = await cursor.fetchall()

        if rows:
            print(f"\n{'Provider':<12} {'Model':<28} {'Requests':>8} {'In Tokens':>10} "
                  f"{'Out Tokens':>11} {'Cost ($)':>10} {'Avg Time':>10}")
            print("-" * 100)
            for row in rows:
                provider, model, requests, in_tokens, out_tokens, cost, avg_time = row
                print(f"{provider:<12} {model:<28} {requests:>8} {in_tokens or 0:>10} "
                      f"{out_tokens or 0:>11} {cost or 0:>10.6f} {avg_time or 0:>9.2f}s")

        # Total cost
        cursor = await db.execute(
            "SELECT SUM(total_cost) FROM llm_requests WHERE status = 'success'"
        )
        total = await cursor.fetchone()
        total_cost = total[0] if total[0] else 0
        print("-" * 100)
        print(f"{'TOTAL':<12} {'':<28} {'':<8} {'':<10} {'':<11} {total_cost:>10.6f}")

        # Error count
        cursor = await db.execute(
            "SELECT COUNT(*) FROM llm_requests WHERE status = 'error'"
        )
        error_count = (await cursor.fetchone())[0]
        if error_count > 0:
            print(f"\nErrors: {error_count} request(s) failed")


async def main() -> None:
    """Run the demo."""
    print("=" * 80)
    print("majordomo-llm Demo: Multi-Provider LLM with Request Logging")
    print("=" * 80)
    print()

    # Check available providers
    available_providers = check_api_keys()
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
                    llm = get_llm_instance(provider, model)
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
        await print_summary(DB_PATH)

    finally:
        await db.close()
        await storage.close()

    print(f"\nDone! Check {STORAGE_DIR} for request/response JSON files.")


if __name__ == "__main__":
    asyncio.run(main())
