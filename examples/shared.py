"""Shared utilities for example scripts."""

import os
from pathlib import Path

import aiosqlite
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# One model per provider with their API key environment variable
PROVIDERS = [
    ("openai", "gpt-4.1-mini", "OPENAI_API_KEY"),
    ("anthropic", "claude-3-5-haiku-latest", "ANTHROPIC_API_KEY"),
    ("gemini", "gemini-2.0-flash", "GEMINI_API_KEY"),
    ("deepseek", "deepseek-chat", "DEEPSEEK_API_KEY"),
    ("cohere", "command-r-08-2024", "CO_API_KEY"),
]

# Base directory for examples
EXAMPLES_DIR = Path(__file__).parent


def get_available_providers() -> list[tuple[str, str]]:
    """Get all providers with API keys configured.

    Returns:
        List of (provider, model) tuples for providers with valid API keys.
    """
    available = []
    missing = []
    for provider, model, env_var in PROVIDERS:
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
