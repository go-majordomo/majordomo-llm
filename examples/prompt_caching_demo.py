#!/usr/bin/env python3
"""Demo script showcasing prompt caching flows.

Two flavors of prompt caching are demonstrated, side by side:

1. EXPLICIT caching (Anthropic, Bedrock Mantle) — this library stamps an
   ephemeral ``cache_control`` breakpoint on the system prompt, so it controls
   cache CREATION. These providers report ``cache_creation_tokens`` (the write)
   and ``cached_tokens`` (the read) separately, and the ``use_prompt_caching``
   toggle on ``get_llm_instance`` turns the breakpoint on or off.

2. AUTOMATIC caching (OpenAI, Gemini, DeepSeek) — the provider caches repeated
   prompt prefixes server-side with no breakpoint to set. There is no creation
   step and nothing to toggle; ``cached_tokens`` simply populate on a repeat and
   are billed at the discounted ``cached_input_cost`` rate.

Each flow uses a large, reused system prompt as the cacheable prefix. Both show
how ``input_cost`` reflects the cache: additive (read/write added on top) for
the explicit providers, subset (cached reads re-priced down) for the automatic
ones. Rates come from ``cached_input_cost`` / ``cache_write_cost`` in
``llm_config.yaml``.

Note: a prefix is only cached once it exceeds a provider/model minimum. Claude
Opus/Sonnet cache from ~1024 tokens, but Claude Haiku needs a much larger prefix
(its minimum is higher AND its tokenizer emits fewer tokens for the same text),
so this demo sends a deliberately large system prompt to clear Haiku too. Cache
hits are otherwise best-effort and time-bounded: OpenAI automatic caching is
eventually consistent (a back-to-back call may miss), Gemini implicit caching is
not guaranteed, and Anthropic's ephemeral cache lives ~5 minutes. Routing
through ``--gateway`` can also lower the hit rate versus calling providers
directly, so an occasional 0-token read is expected.

Prerequisites:
    1. Install dependencies: uv sync --all-extras
    2. Set at least one API key (see the two provider groups below).

Usage:
    uv run python examples/prompt_caching_demo.py            # call providers directly
    uv run python examples/prompt_caching_demo.py --gateway  # route through Steward
    uv run python examples/prompt_caching_demo.py --provider anthropic

Gateway routing (--gateway) reads:
    - MAJORDOMO_GATEWAY_URL (defaults to http://localhost:7680)
    - MAJORDOMO_API_KEY (required)
"""

import os

from shared import gateway_kwargs, run_demo

from majordomo_llm import get_llm_instance
from majordomo_llm.base import TOKENS_PER_MILLION

# (provider, model, required_env_vars, explicit_cache_control).
# explicit_cache_control=True → this library controls the cache breakpoint, so
# the creation + use_prompt_caching-toggle flows apply. False → the provider
# caches automatically and only the read flow is observable.
PROVIDERS: list[tuple[str, str, tuple[str, ...], bool]] = [
    # Explicit cache-control providers — current flagships.
    ("anthropic", "claude-sonnet-5", ("ANTHROPIC_API_KEY",), True),
    ("anthropic", "claude-opus-4-8-fast", ("ANTHROPIC_API_KEY",), True),
    (
        "bedrock_mantle",
        "anthropic.claude-haiku-4-5",
        ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION"),
        True,
    ),
    # Automatic-caching providers — current flagships.
    ("openai", "gpt-5.6-luna", ("OPENAI_API_KEY",), False),
    ("gemini", "gemini-3.6-flash", ("GEMINI_API_KEY",), False),
    ("deepseek", "deepseek-v4-flash", ("DEEPSEEK_API_KEY",), False),
]

# Two different user turns so each call does real work while the large system
# prompt (the cacheable prefix) stays identical across them.
FIRST_TURN = "In one sentence, summarize your role."
SECOND_TURN = "Now list your top three operating priorities as a bulleted list."


def build_large_system_prompt(min_chars: int = 24000) -> str:
    """Build a system prompt large enough to exceed provider cache thresholds.

    Claude Opus/Sonnet cache from ~1024 tokens, but Claude Haiku 4.5 needs more:
    empirically a ~2560-token prefix does NOT cache on Haiku (direct or via
    Bedrock Mantle) while it caches fine on Sonnet/Opus. Haiku also tokenizes the
    same text to fewer tokens, so a generous ~24K characters (~5K Haiku tokens)
    is used to clear its minimum on every model in the sweep.
    """
    header = (
        "You are the senior support engineer for the Acme Cloud Platform. "
        "Answer strictly according to the operating procedures below.\n\n"
    )
    procedure = (
        "Verify the customer's account tier before quoting any SLA. Escalate "
        "security incidents to the on-call responder within five minutes and "
        "never expose internal host names, credentials, or region identifiers "
        "in a reply. Prefer the least-privilege remediation and always record "
        "the ticket ID in your summary."
    )
    prompt = header
    n = 1
    while len(prompt) < min_chars:
        prompt += f"Procedure {n}. {procedure}\n"
        n += 1
    return prompt


def _print_usage(label: str, response: object) -> None:
    """Print the cache token + cost breakdown for one response."""
    print(f"    {label}")
    print(
        f"      Tokens: {response.input_tokens} in / {response.output_tokens} out "
        f"| cache write {response.cache_creation_tokens} / "
        f"cache read {response.cached_tokens}"
    )
    print(
        f"      Cost: ${response.total_cost:.6f} "
        f"(input ${response.input_cost:.6f} + output ${response.output_cost:.6f})"
    )


def _print_savings(llm: object, cached_tokens: int) -> None:
    """Print the prompt-side saving from a cache read, if any rate is configured.

    The per-token saving is the same in both accounting modes: the gap between
    the full input rate and the discounted cache-read rate.
    """
    if cached_tokens > 0 and llm.cached_input_cost is not None:
        saved = cached_tokens * (llm.input_cost - llm.cached_input_cost) / TOKENS_PER_MILLION
        print(
            f"      Cache hit: {cached_tokens} tokens read; "
            f"prompt-side savings vs. uncached ~= ${saved:.6f}"
        )


async def demo_caching(
    provider: str, model: str, explicit: bool, use_gateway: bool = False
) -> bool:
    """Run the read flow (both groups) plus the creation/toggle flow (explicit)."""
    system_prompt = build_large_system_prompt()
    kind = "explicit cache-control" if explicit else "automatic caching"
    print(f"\n  [{provider}/{model}]  ({kind}), system prompt ~{len(system_prompt)} chars")

    try:
        llm = get_llm_instance(
            provider,
            model,
            **gateway_kwargs(use_gateway, feature="prompt-caching-demo"),
        )

        # Flow A — cold then warm. The first call primes the cache (a write on
        # explicit providers); the second reuses the prefix and reads it back.
        write_hint = "cache WRITE > 0" if explicit else "cache read 0"
        print("\n    Flow A — reuse the same system prompt across two calls:")
        first = await llm.get_response(FIRST_TURN, system_prompt=system_prompt)
        _print_usage(f"Call 1 (cold — expect {write_hint}):", first)

        second = await llm.get_response(SECOND_TURN, system_prompt=system_prompt)
        _print_usage("Call 2 (warm — expect cache READ > 0):", second)
        _print_savings(llm, second.cached_tokens)

        # Flow B — only meaningful where this library controls the breakpoint.
        if explicit:
            print("\n    Flow B — caching OFF (use_prompt_caching=False):")
            uncached_llm = get_llm_instance(
                provider,
                model,
                use_prompt_caching=False,
                **gateway_kwargs(use_gateway, feature="prompt-caching-demo"),
            )
            uncached = await uncached_llm.get_response(
                FIRST_TURN, system_prompt=system_prompt
            )
            _print_usage("Call (expect cache write/read == 0):", uncached)
        else:
            print(
                "\n    (No use_prompt_caching toggle — this provider caches "
                "automatically with no breakpoint to disable.)"
            )
    except Exception as e:
        print(f"    Error: {e}")
        return False

    return True


async def main(use_gateway: bool = False, provider: str | None = None) -> None:
    print("=" * 80)
    print("majordomo-llm: Prompt Caching Demo")
    print("=" * 80)

    if use_gateway:
        print("Routing through the Majordomo gateway (Steward).")

    entries = PROVIDERS if provider is None else [e for e in PROVIDERS if e[0] == provider]
    if provider is not None and not entries:
        known = ", ".join(sorted({p for p, _, _, _ in PROVIDERS}))
        print(f"Unknown provider '{provider}'. Known providers: {known}")
        return

    available = [
        (p, m, explicit)
        for p, m, env, explicit in entries
        if all(os.environ.get(var) for var in env)
    ]
    if not available:
        print("No usable credentials found. Set one of:")
        print("  ANTHROPIC_API_KEY | AWS_BEARER_TOKEN_BEDROCK + AWS_REGION")
        print("  OPENAI_API_KEY | GEMINI_API_KEY | DEEPSEEK_API_KEY")
        return

    print(f"Available providers: {', '.join(p for p, _, _ in available)}")

    failures = 0
    for prov, model, explicit in available:
        ok = await demo_caching(prov, model, explicit, use_gateway=use_gateway)
        if not ok:
            failures += 1

    print("\n" + "=" * 80)
    total = len(available)
    print(f"Done! {total - failures}/{total} succeeded.")


if __name__ == "__main__":
    run_demo(main, description=__doc__)
