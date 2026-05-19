"""Smoke-test majordomo-llm across providers, direct and through Steward.

Two passes:

  Pass 1 — provider capability matrix: one canonical (latest) model per provider,
           exercises text / JSON / structured / stream. Catches provider-class bugs
           in majordomo-llm and in Steward's per-provider translation.

  Pass 2 — per-model smoke: additional representative models per provider,
           exercises text + stream only. Catches "is this model name routable
           through Steward" bugs (the Opus 4.7 class of breakage).

Each cell is run twice: once direct, once through Steward. A diff between the two
isolates Steward bugs from provider/library bugs.

Environment:
  MAJORDOMO_GATEWAY_URL    Steward base URL (default: http://localhost:7680)
  MAJORDOMO_API_KEY        Steward API key (required for the steward leg)
  OPENAI_API_KEY           Required if openai is in scope
  ANTHROPIC_API_KEY        Required if anthropic is in scope
  GEMINI_API_KEY           Required if gemini is in scope
  DEEPSEEK_API_KEY         Required if deepseek is in scope
  CO_API_KEY               Required if cohere is in scope
  AWS_BEARER_TOKEN_BEDROCK Bedrock auto-skips if unset

Usage:
  uv run python scripts/smoke_test_providers.py
  uv run python scripts/smoke_test_providers.py --provider anthropic
  uv run python scripts/smoke_test_providers.py --capability stream
  uv run python scripts/smoke_test_providers.py --skip-direct
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic import BaseModel

from majordomo_llm import get_llm_instance
from majordomo_llm.base import LLM
from majordomo_llm.exceptions import StructuredOutputUnsupported
from majordomo_llm.factory import LLM_CONFIG, get_supported_providers

load_dotenv()

PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "cohere": "CO_API_KEY",
    "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
}

# Providers currently routable through Steward. Others run direct-only — their
# steward-leg rows are suppressed so they don't pollute the matrix with known
# "not yet supported" failures.
STEWARD_SUPPORTED_PROVIDERS: set[str] = {"openai", "anthropic", "gemini"}

# Per-provider additional models for Pass 2 (text + stream only).
# Pass 1's canonical model is auto-picked as the first entry in llm_config.yaml.
EXTRA_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-5.4", "gpt-5.4-mini"],
    "anthropic": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "gemini": ["gemini-3-flash-preview", "gemini-3.1-flash-lite-preview"],
    "deepseek": ["deepseek-v4-flash"],
    "cohere": [],
    "bedrock": [],
}

OK = "✓"
FAIL = "✗"
SKIP = "—"


class _Person(BaseModel):
    name: str
    age: int


@dataclass
class CellResult:
    status: str  # OK, FAIL, SKIP
    elapsed: float = 0.0
    error: str = ""  # Truncated, for inline matrix display.
    full_error: str = ""  # Full exception text, written to the sidecar log.


async def _run_text(llm: LLM, extra_headers: dict[str, str] | None) -> CellResult:
    t = time.time()
    r = await llm.get_response(
        "Reply with just the word OK.", temperature=0.0, extra_headers=extra_headers,
    )
    ok = bool(r.content and r.content.strip())
    return CellResult(OK if ok else FAIL, time.time() - t,
                      "" if ok else "empty content")


async def _run_json(llm: LLM, extra_headers: dict[str, str] | None) -> CellResult:
    t = time.time()
    r = await llm.get_json_response(
        'Reply with the JSON object {"status": "ok"} and nothing else.',
        temperature=0.0,
        extra_headers=extra_headers,
    )
    ok = isinstance(r.content, dict) and "status" in r.content
    return CellResult(OK if ok else FAIL, time.time() - t,
                      "" if ok else f"unexpected payload: {r.content!r}")


async def _run_structured(llm: LLM, extra_headers: dict[str, str] | None) -> CellResult:
    t = time.time()
    try:
        r = await llm.get_structured_json_response(
            response_model=_Person,
            user_prompt="Extract the person: Alice is 30 years old.",
            temperature=0.0,
            extra_headers=extra_headers,
        )
    except StructuredOutputUnsupported:
        return CellResult(SKIP, 0.0, "structured output unsupported")
    ok = isinstance(r.content, _Person) and r.content.age == 30
    return CellResult(OK if ok else FAIL, time.time() - t,
                      "" if ok else f"unexpected model: {r.content!r}")


async def _run_stream(llm: LLM, extra_headers: dict[str, str] | None) -> CellResult:
    t = time.time()
    stream = await llm.get_response_stream(
        "Reply with just the word OK.", temperature=0.0, extra_headers=extra_headers,
    )
    chunks: list[str] = []
    async for chunk in stream:
        chunks.append(chunk)
    text = "".join(chunks)
    ok = bool(text.strip())
    return CellResult(OK if ok else FAIL, time.time() - t,
                      "" if ok else "empty stream")


CapabilityFn = Callable[[LLM, "dict[str, str] | None"], Awaitable[CellResult]]

CAPABILITIES: dict[str, CapabilityFn] = {
    "text": _run_text,
    "json": _run_json,
    "structured": _run_structured,
    "stream": _run_stream,
}


def _canonical_model(provider: str) -> str:
    """First model in llm_config.yaml for the provider (= latest by convention)."""
    return next(iter(LLM_CONFIG[provider]["models"]))


def _steward_default_headers(gateway_key: str, run_id: str) -> dict[str, str]:
    return {
        "X-Majordomo-Key": gateway_key,
        "X-Majordomo-Feature": "smoke-test",
        "X-Majordomo-Project": "majordomo-llm",
        "X-Majordomo-Run-Id": run_id,
    }


def _steward_base_url(provider: str, gateway_url: str) -> str:
    """Per-provider base URL adjustments for routing through Steward.

    The OpenAI SDK assumes its base_url already includes the ``/v1`` path
    segment (its default is ``https://api.openai.com/v1``), so when we point it
    at a bare gateway URL the SDK constructs paths like ``/responses`` instead
    of ``/v1/responses`` and Steward rejects them. Append ``/v1`` for OpenAI
    to match what the SDK expects.
    """
    base = gateway_url.rstrip("/")
    if provider == "openai":
        return f"{base}/v1"
    return base


def _build_llm(
    provider: str,
    model: str,
    *,
    via_steward: bool,
    gateway_url: str,
    gateway_key: str | None,
    run_id: str,
) -> LLM:
    if via_steward:
        assert gateway_key is not None
        return get_llm_instance(
            provider,
            model,
            base_url=_steward_base_url(provider, gateway_url),
            default_headers=_steward_default_headers(gateway_key, run_id),
        )
    return get_llm_instance(provider, model)


@dataclass
class Row:
    provider: str
    model: str
    route: str  # "direct" or "steward"
    cells: dict[str, CellResult] = field(default_factory=dict)


async def _run_cell(
    provider: str,
    model: str,
    capability: str,
    pass_name: str,
    *,
    via_steward: bool,
    gateway_url: str,
    gateway_key: str | None,
    run_id: str,
) -> CellResult:
    try:
        llm = _build_llm(
            provider, model,
            via_steward=via_steward,
            gateway_url=gateway_url,
            gateway_key=gateway_key,
            run_id=run_id,
        )
        # Per-call headers only matter on the steward leg; direct providers
        # would just ignore them but we keep the wire clean.
        extra_headers: dict[str, str] | None = None
        if via_steward:
            extra_headers = {
                "X-Majordomo-Capability": capability,
                "X-Majordomo-Pass": pass_name,
            }
        return await CAPABILITIES[capability](llm, extra_headers)
    except Exception as e:  # noqa: BLE001 — smoke test wants to keep going
        full = f"{type(e).__name__}: {e}"
        return CellResult(
            status=FAIL,
            elapsed=0.0,
            error=full[:200].replace("\n", " "),
            full_error=full,
        )


def _print_row(row: Row, caps: list[str]) -> None:
    cells = " ".join(f"{cap}:{row.cells[cap].status}" for cap in caps)
    print(f"  [{row.route:7}] {row.provider:9} {row.model:42} {cells}")


async def _run_all(
    providers: list[str],
    capabilities: list[str],
    routes: list[tuple[str, bool]],
    gateway_url: str,
    gateway_key: str | None,
    run_id: str,
) -> int:
    all_rows: list[Row] = []

    def routes_for(provider: str) -> list[tuple[str, bool]]:
        return [
            (name, via) for name, via in routes
            if not via or provider in STEWARD_SUPPORTED_PROVIDERS
        ]

    pass1_name = "capability-matrix"
    print("=" * 90)
    print(f"Pass 1 ({pass1_name}) — canonical model per provider × all capabilities")
    print("=" * 90)
    for provider in providers:
        model = _canonical_model(provider)
        for route_name, via_steward in routes_for(provider):
            row = Row(provider=provider, model=model, route=route_name)
            for cap in capabilities:
                row.cells[cap] = await _run_cell(
                    provider, model, cap, pass1_name,
                    via_steward=via_steward,
                    gateway_url=gateway_url,
                    gateway_key=gateway_key,
                    run_id=run_id,
                )
            all_rows.append(row)
            _print_row(row, capabilities)

    pass2_caps = [c for c in ("text", "stream") if c in capabilities]
    pass2_name = "per-model-smoke"
    print()
    print("=" * 90)
    print(f"Pass 2 ({pass2_name}) — extra models × text + stream")
    print("=" * 90)
    if not pass2_caps:
        print("  (skipped — neither text nor stream in --capability filter)")
    else:
        any_pass2 = False
        for provider in providers:
            for model in EXTRA_MODELS.get(provider, []):
                any_pass2 = True
                for route_name, via_steward in routes_for(provider):
                    row = Row(provider=provider, model=model, route=route_name)
                    for cap in pass2_caps:
                        row.cells[cap] = await _run_cell(
                            provider, model, cap, pass2_name,
                            via_steward=via_steward,
                            gateway_url=gateway_url,
                            gateway_key=gateway_key,
                            run_id=run_id,
                        )
                    all_rows.append(row)
                    _print_row(row, pass2_caps)
        if not any_pass2:
            print("  (no extra models configured for selected providers)")

    print()
    print("=" * 90)
    print("Failures")
    print("=" * 90)
    fail_count = 0
    log_path = f"smoke-test-{run_id}.log"
    log_lines: list[str] = []
    for row in all_rows:
        for cap, cell in row.cells.items():
            if cell.status == FAIL:
                fail_count += 1
                print(f"  [{row.route:7}] {row.provider}/{row.model} {cap}: {cell.error}")
                log_lines.append(
                    f"[{row.route}] {row.provider}/{row.model} {cap}\n"
                    f"{cell.full_error}\n"
                    f"{'-' * 80}\n"
                )
    if fail_count == 0:
        print("  (none)")
    elif log_lines:
        with open(log_path, "w") as f:
            f.write(f"Smoke-test run {run_id}\n{'=' * 80}\n\n")
            f.writelines(log_lines)
        print()
        print(f"  Full error bodies written to: {log_path}")

    # Highlight steward-vs-direct divergences — these are the most actionable signal.
    print()
    print("=" * 90)
    print("Steward regressions (cells that passed direct but failed through Steward)")
    print("=" * 90)
    by_key: dict[tuple[str, str, str], dict[str, CellResult]] = {}
    for row in all_rows:
        for cap, cell in row.cells.items():
            by_key.setdefault((row.provider, row.model, cap), {})[row.route] = cell
    divergence_count = 0
    for (provider, model, cap), routes_map in sorted(by_key.items()):
        direct = routes_map.get("direct")
        steward = routes_map.get("steward")
        if direct and steward and direct.status == OK and steward.status == FAIL:
            divergence_count += 1
            print(f"  {provider}/{model} {cap}: {steward.error}")
    if divergence_count == 0:
        print("  (none)")

    return fail_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider", action="append", default=None,
        choices=get_supported_providers(),
        help="Restrict to one provider (repeatable).",
    )
    parser.add_argument(
        "--capability", action="append", default=None,
        choices=list(CAPABILITIES.keys()),
        help="Restrict to one capability (repeatable).",
    )
    parser.add_argument(
        "--skip-direct", action="store_true",
        help="Only run through-steward calls.",
    )
    parser.add_argument(
        "--skip-steward", action="store_true",
        help="Only run direct calls.",
    )
    args = parser.parse_args()

    if args.skip_direct and args.skip_steward:
        print("ERROR: cannot pass both --skip-direct and --skip-steward.", file=sys.stderr)
        return 2

    gateway_url = os.environ.get("MAJORDOMO_GATEWAY_URL", "http://localhost:7680")
    gateway_key = os.environ.get("MAJORDOMO_API_KEY")
    if not args.skip_steward and not gateway_key:
        print(
            "ERROR: MAJORDOMO_API_KEY not set (required for steward leg). "
            "Pass --skip-steward to run only direct calls.",
            file=sys.stderr,
        )
        return 2

    requested_providers = args.provider or get_supported_providers()
    capabilities = args.capability or list(CAPABILITIES.keys())

    runnable_providers: list[str] = []
    for provider in requested_providers:
        env_var = PROVIDER_API_KEY_ENV[provider]
        if os.environ.get(env_var):
            runnable_providers.append(provider)
            continue
        if provider == "bedrock":
            print(f"[skip] bedrock: {env_var} not set", file=sys.stderr)
            continue
        print(
            f"ERROR: {env_var} not set (required for provider {provider!r}). "
            f"Set the env var or pass --provider to exclude it.",
            file=sys.stderr,
        )
        return 2

    if not runnable_providers:
        print("ERROR: no runnable providers.", file=sys.stderr)
        return 2

    routes: list[tuple[str, bool]] = []
    if not args.skip_direct:
        routes.append(("direct", False))
    if not args.skip_steward:
        routes.append(("steward", True))

    run_id = str(uuid.uuid4())
    steward_only_direct = [
        p for p in runnable_providers if p not in STEWARD_SUPPORTED_PROVIDERS
    ]
    print(f"Run ID: {run_id}")
    print(f"Gateway: {gateway_url}")
    print(f"Providers: {', '.join(runnable_providers)}")
    print(f"Capabilities: {', '.join(capabilities)}")
    print(f"Routes: {', '.join(name for name, _ in routes)}")
    if steward_only_direct and not args.skip_steward:
        print(
            f"Direct-only (not yet routable via Steward): "
            f"{', '.join(steward_only_direct)}"
        )
    print()

    fail_count = asyncio.run(_run_all(
        runnable_providers, capabilities, routes, gateway_url, gateway_key, run_id,
    ))
    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
