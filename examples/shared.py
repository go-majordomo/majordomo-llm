"""Shared utilities for example scripts."""

import argparse
import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import aiosqlite
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()


def run_demo(
    main: Callable[..., Awaitable[None]], description: str | None = None
) -> None:
    """Parse the shared ``--gateway`` / ``--provider`` CLI flags and run a demo.

    Every example exposes the same two flags and a ``main(use_gateway, provider)``
    coroutine, so the argparse boilerplate lives here rather than being copied
    into each script's ``__main__`` block.

    Args:
        main: The demo's async entry point, accepting ``use_gateway`` (bool) and
            ``provider`` (str | None) keyword arguments.
        description: Help text for ``--help`` — typically the module ``__doc__``.
    """
    parser = argparse.ArgumentParser(description=description)
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
    args = parser.parse_args()
    asyncio.run(main(use_gateway=args.gateway, provider=args.provider))

# Provider/model pairs with their required environment variables.
# Each entry is (provider, model, (env_var, ...)) — all listed env vars must
# be set for the entry to be selected.
PROVIDERS: list[tuple[str, str, tuple[str, ...]]] = [
    # Native provider SDKs — the latest fast/small model from each, plus a
    # low-effort Opus 4.8 profile so a frontier model appears in the sweep.
    ("openai", "gpt-5.6-luna", ("OPENAI_API_KEY",)),
    ("anthropic", "claude-haiku-4-5-20251001", ("ANTHROPIC_API_KEY",)),
    ("anthropic", "claude-opus-4-8-fast", ("ANTHROPIC_API_KEY",)),
    ("gemini", "gemini-3.6-flash", ("GEMINI_API_KEY",)),
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
    (
        "bedrock",
        "us.meta.llama4-scout-17b-instruct-v1:0",
        ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION"),
    ),
    ("bedrock", "deepseek.v3.2", ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION")),
    # OpenAI-compatible serverless hosts — using the cheapest small model from each.
    ("fireworks", "accounts/fireworks/models/kimi-k2p6", ("FIREWORKS_API_KEY",)),
    ("together", "deepseek-ai/DeepSeek-V4-Pro", ("TOGETHER_API_KEY",)),
]

# Base directory for examples
EXAMPLES_DIR = Path(__file__).parent

# Default Steward endpoint when MAJORDOMO_GATEWAY_URL is not set (self-hosted).
DEFAULT_GATEWAY_URL = "http://localhost:7680"

# Providers whose native wire format Steward does not accept, so they cannot be
# routed through the gateway (they still work in direct mode). Cohere speaks its
# own /v2/chat shape rather than the OpenAI/Anthropic/Gemini/Bedrock shapes the
# gateway understands.
GATEWAY_UNSUPPORTED_PROVIDERS = frozenset({"cohere"})


def gateway_kwargs(use_gateway: bool, feature: str | None = None) -> dict:
    """Build get_llm_instance kwargs for routing through the Majordomo gateway.

    When routing through Steward, the provider API key is picked up from the
    environment by the gateway itself; callers pass the Steward URL as base_url
    and authenticate to Steward with the X-Majordomo-Key header.

    Args:
        use_gateway: When False, returns an empty dict so callers hit providers
            directly (the unchanged default behavior).
        feature: Optional value for the X-Majordomo-Feature header, which becomes
            a filterable cost-attribution dimension in the Majordomo dashboard.

    Returns:
        Kwargs (base_url, default_headers) to splat into get_llm_instance, or an
        empty dict when use_gateway is False.

    Raises:
        RuntimeError: If use_gateway is True but MAJORDOMO_API_KEY is unset.
    """
    if not use_gateway:
        return {}

    api_key = os.environ.get("MAJORDOMO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MAJORDOMO_API_KEY must be set to route through the Majordomo gateway."
        )

    base_url = os.environ.get("MAJORDOMO_GATEWAY_URL", DEFAULT_GATEWAY_URL)
    headers = {"X-Majordomo-Key": api_key}
    if feature:
        headers["X-Majordomo-Feature"] = feature

    return {"base_url": base_url, "default_headers": headers}


def get_available_providers(
    use_gateway: bool = False, provider: str | None = None
) -> list[tuple[str, str]]:
    """Get all providers with API keys configured.

    Args:
        use_gateway: When True, drop providers whose wire format Steward cannot
            route (see GATEWAY_UNSUPPORTED_PROVIDERS) so gateway runs don't hit
            a guaranteed failure.
        provider: When set, restrict to entries for this provider only. An
            unknown provider name prints the known set and returns an empty list.

    Returns:
        List of (provider, model) tuples for providers whose required
        environment variables are all set.
    """
    entries = PROVIDERS
    if provider is not None:
        entries = [e for e in PROVIDERS if e[0] == provider]
        if not entries:
            known = ", ".join(sorted({p for p, _, _ in PROVIDERS}))
            print(f"Unknown provider '{provider}'. Known providers: {known}\n")
            return []

    available = []
    missing = []
    gateway_skipped = []
    for provider_name, model, env_vars in entries:
        unset = [var for var in env_vars if not os.environ.get(var)]
        if unset:
            missing.append((provider_name, model, unset))
        elif use_gateway and provider_name in GATEWAY_UNSUPPORTED_PROVIDERS:
            gateway_skipped.append((provider_name, model))
        else:
            available.append((provider_name, model))

    if missing:
        print("Missing environment variables (these entries will be skipped):")
        for provider_name, model, unset in missing:
            print(f"  - {provider_name}:{model} requires {', '.join(unset)}")
        print()

    if gateway_skipped:
        print("Not routable through the gateway (these entries will be skipped):")
        for provider_name, model in gateway_skipped:
            print(f"  - {provider_name}:{model} — Steward does not support its wire format")
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
