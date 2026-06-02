"""Shared utilities for example scripts."""

import os
from pathlib import Path

import aiosqlite
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Provider/model pairs with their required environment variables.
# Each entry is (provider, model, (env_var, ...)) — all listed env vars must
# be set for the entry to be selected.
PROVIDERS: list[tuple[str, str, tuple[str, ...]]] = [
    # Native provider SDKs — using the latest fast/small model from each.
    ("openai", "gpt-5.4-mini", ("OPENAI_API_KEY",)),
    ("anthropic", "claude-haiku-4-5-20251001", ("ANTHROPIC_API_KEY",)),
    ("gemini", "gemini-3-flash-preview", ("GEMINI_API_KEY",)),
    ("deepseek", "deepseek-v4-flash", ("DEEPSEEK_API_KEY",)),
    ("cohere", "command-a-03-2025", ("CO_API_KEY",)),
    # Amazon Bedrock Mantle — Claude via AWS-native Anthropic Messages API.
    (
        "bedrock_mantle",
        "anthropic.claude-haiku-4-5",
        ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION"),
    ),
    # Amazon Bedrock (Converse API) — non-Anthropic upstream providers.
    ("bedrock", "moonshotai.kimi-k2.5", ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION")),
    ("bedrock", "nvidia.nemotron-nano-12b-v2", ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION")),
    (
        "bedrock",
        "us.meta.llama4-scout-17b-instruct-v1:0",
        ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION"),
    ),
    ("bedrock", "deepseek.v3.2", ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION")),
    # OpenAI-compatible serverless hosts — using the cheapest small model from each.
    ("fireworks", "accounts/fireworks/models/kimi-k2p5", ("FIREWORKS_API_KEY",)),
    ("together", "deepseek-ai/DeepSeek-V4-Pro", ("TOGETHER_API_KEY",)),
]

# Base directory for examples
EXAMPLES_DIR = Path(__file__).parent


def get_available_providers() -> list[tuple[str, str]]:
    """Get all providers with API keys configured.

    Returns:
        List of (provider, model) tuples for providers whose required
        environment variables are all set.
    """
    available = []
    missing = []
    for provider, model, env_vars in PROVIDERS:
        unset = [var for var in env_vars if not os.environ.get(var)]
        if not unset:
            available.append((provider, model))
        else:
            missing.append((provider, model, unset))

    if missing:
        print("Missing environment variables (these entries will be skipped):")
        for provider, model, unset in missing:
            print(f"  - {provider}:{model} requires {', '.join(unset)}")
        print()

    return available


async def clear_database(db_path: Path, storage_dir: Path) -> None:
    """Clear the database and storage directory before running demos.

    Args:
        db_path: Path to the SQLite database file.
        storage_dir: Path to the storage directory for request/response bodies.
    """
    if db_path.exists():
        db_path.unlink()
    if storage_dir.exists():
        for f in storage_dir.glob("*"):
            f.unlink()


async def print_summary(db_path: Path, title: str = "COST SUMMARY") -> None:
    """Query SQLite and print a summary of all requests.

    Args:
        db_path: Path to the SQLite database file.
        title: Title to display in the summary header.
    """
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

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
            print(
                f"\n{'Provider':<12} {'Model':<28} {'Requests':>8} {'In Tokens':>10} "
                f"{'Out Tokens':>11} {'Cost ($)':>10} {'Avg Time':>10}"
            )
            print("-" * 100)
            for row in rows:
                provider, model, requests, in_tokens, out_tokens, cost, avg_time = row
                print(
                    f"{provider:<12} {model:<28} {requests:>8} {in_tokens or 0:>10} "
                    f"{out_tokens or 0:>11} {cost or 0:>10.6f} {avg_time or 0:>9.2f}s"
                )

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
