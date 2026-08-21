#!/usr/bin/env python3
"""Demo script comparing one open-weight model across every host that serves it.

The other demos are provider-major: pick a provider, run its model. This one is
model-major. It fixes the *weights* and fans out across hosts, which is the
question this library exists to answer — for identical output quality, what does
each host charge and how fast is it?

The spread is real. GLM-5.2 input runs $0.75/M on DeepInfra and $1.40/M on four
other hosts; cached reads run $0.14/M on Baseten and $0.26/M on Novita, while
Nebius publishes no cache tier at all. Same weights, ~2x price range.

This script demonstrates:

- Running identical prompts against the same model on every configured host
- Per-host cost, latency, and tokens, ranked cheapest-first
- Configured rates pulled from llm_config.yaml via get_model_pricing(), so the
  table shows what you are billed as well as what you paid on this call

Prerequisites:
    1. Install dependencies: uv sync --all-extras
    2. Set one or more of:
       - FIREWORKS_API_KEY, TOGETHER_API_KEY, BASETEN_API_KEY,
         DEEPINFRA_API_KEY, NOVITA_API_KEY, NEBIUS_API_KEY, MOONSHOT_API_KEY

    Hosts without a key are skipped, so a single key still produces useful output.

Gateway mode (--gateway) routes through Majordomo Steward:
    - MAJORDOMO_GATEWAY_URL (defaults to http://localhost:7680)
    - MAJORDOMO_API_KEY (required)
"""

import os
import textwrap
import time

from shared import gateway_kwargs, run_demo, unavailable_provider_message

from majordomo_llm import get_llm_instance, get_model_pricing
from majordomo_llm.base import TOKENS_PER_MILLION

# A fixed reference workload for rate-only comparison. Ranking hosts by what one
# live call happened to cost is misleading: a chattier host emits more output
# tokens and looks expensive even at a cheaper rate. Pricing an IDENTICAL
# hypothetical usage from llm_config.yaml isolates the rate.
REFERENCE_INPUT_TOKENS = 10_000
REFERENCE_OUTPUT_TOKENS = 1_000

# Canonical model -> the hosts that serve it, with each host's own model ID.
# Model IDs are NOT interchangeable across hosts: casing and org prefixes differ
# (Novita uses deepseek/deepseek-v4-pro where the HF-style hosts use
# deepseek-ai/DeepSeek-V4-Pro, and Baseten lowercases thinkingmachines/inkling).
MODEL_FAMILIES: list[tuple[str, list[tuple[str, str, tuple[str, ...]]]]] = [
    (
        "GLM-5.2",
        [
            ("fireworks", "accounts/fireworks/models/glm-5p2", ("FIREWORKS_API_KEY",)),
            ("together", "zai-org/GLM-5.2", ("TOGETHER_API_KEY",)),
            ("baseten", "zai-org/GLM-5.2", ("BASETEN_API_KEY",)),
            ("deepinfra", "zai-org/GLM-5.2", ("DEEPINFRA_API_KEY",)),
            ("novita", "zai-org/glm-5.2", ("NOVITA_API_KEY",)),
            ("nebius", "zai-org/GLM-5.2", ("NEBIUS_API_KEY",)),
        ],
    ),
    (
        "Kimi K3",
        [
            ("fireworks", "accounts/fireworks/models/kimi-k3", ("FIREWORKS_API_KEY",)),
            ("together", "moonshotai/Kimi-K3", ("TOGETHER_API_KEY",)),
            ("baseten", "moonshotai/Kimi-K3", ("BASETEN_API_KEY",)),
            ("deepinfra", "moonshotai/Kimi-K3", ("DEEPINFRA_API_KEY",)),
            ("novita", "moonshotai/kimi-k3", ("NOVITA_API_KEY",)),
            ("nebius", "moonshotai/Kimi-K3", ("NEBIUS_API_KEY",)),
            # Moonshot is the first-party vendor for Kimi, so it belongs in this
            # comparison as the reference rate the resellers price against.
            ("moonshot", "kimi-k3", ("MOONSHOT_API_KEY",)),
        ],
    ),
]

PROMPT = "In two sentences, explain what a Mixture-of-Experts model is."
SYSTEM_PROMPT = "Be concise and precise. Do not use bullet points."


def _rate_summary(provider: str, model: str) -> str:
    """Render the configured per-million rates for a (provider, model) pair."""
    pricing = get_model_pricing(provider, model)
    if pricing is None:
        return "unpriced"
    cached = (
        f"{pricing.cached_input_cost:g}"
        if pricing.cached_input_cost is not None
        else "-"
    )
    return f"${pricing.input_cost:g}/{cached}/{pricing.output_cost:g}"


async def run_host(
    provider: str, model: str, use_gateway: bool = False
) -> dict | None:
    """Call one host and return its measured result, or None on failure."""
    try:
        llm = get_llm_instance(
            provider, model, **gateway_kwargs(use_gateway, feature="open-weight-demo")
        )
        start = time.perf_counter()
        response = await llm.get_response(PROMPT, system_prompt=SYSTEM_PROMPT)
        elapsed = time.perf_counter() - start
    except Exception as e:
        print(f"  {provider:<12} {model:<38} error: {e}")
        return None

    print(
        f"  {provider:<12} {model:<38} "
        f"{response.input_tokens:>5} in {response.output_tokens:>5} out  "
        f"${response.total_cost:.6f}  {elapsed:>5.2f}s"
    )
    return {
        "provider": provider,
        "model": model,
        "cost": response.total_cost,
        "elapsed": elapsed,
        "output_tokens": response.output_tokens,
        "content": response.content,
    }


def _reference_cost(provider: str, model: str) -> float | None:
    """Price the fixed reference workload from configured rates."""
    pricing = get_model_pricing(provider, model)
    if pricing is None:
        return None
    return (
        REFERENCE_INPUT_TOKENS * pricing.input_cost
        + REFERENCE_OUTPUT_TOKENS * pricing.output_cost
    ) / TOKENS_PER_MILLION


def _print_ranking(label: str, results: list[dict]) -> None:
    """Rank the hosts that answered, by rate, by observed cost, and by latency."""
    if len(results) < 2:
        return

    # Rate comparison — the like-for-like number. Same tokens for every host.
    priced = [
        (r, _reference_cost(r["provider"], r["model"]))
        for r in results
    ]
    priced = [(r, c) for r, c in priced if c is not None]
    if len(priced) >= 2:
        print(
            f"\n    {label} — rate comparison "
            f"({REFERENCE_INPUT_TOKENS:,} in + {REFERENCE_OUTPUT_TOKENS:,} out, "
            "identical for every host):"
        )
        for i, (r, cost) in enumerate(sorted(priced, key=lambda x: x[1]), start=1):
            print(f"      {i}. {r['provider']:<12} ${cost:.6f}")
        cheap_r, cheap_c = min(priced, key=lambda x: x[1])
        dear_r, dear_c = max(priced, key=lambda x: x[1])
        if cheap_c > 0 and dear_c > cheap_c:
            print(
                f"      {dear_r['provider']} charges {dear_c / cheap_c:.2f}x "
                f"{cheap_r['provider']}'s rate for identical weights."
            )

    # Observed cost — real, but confounded by how verbose each host was.
    print(f"\n    {label} — what this call actually cost (output length varies):")
    for i, r in enumerate(sorted(results, key=lambda x: x["cost"]), start=1):
        print(
            f"      {i}. {r['provider']:<12} ${r['cost']:.6f}"
            f"   ({r['output_tokens']} output tokens)"
        )

    print(f"\n    {label} — ranked by latency:")
    for i, r in enumerate(sorted(results, key=lambda x: x["elapsed"]), start=1):
        print(f"      {i}. {r['provider']:<12} {r['elapsed']:.2f}s")


async def main(use_gateway: bool = False, provider: str | None = None) -> None:
    print("=" * 100)
    print("majordomo-llm: Open-Weight Cross-Host Comparison")
    print("=" * 100)

    if use_gateway:
        print("Routing through the Majordomo gateway (Steward).")

    known = {p for _, hosts in MODEL_FAMILIES for p, _, _ in hosts}
    if provider is not None and provider not in known:
        print(unavailable_provider_message(provider, known))
        return

    ran_anything = False
    for label, hosts in MODEL_FAMILIES:
        entries = [h for h in hosts if provider is None or h[0] == provider]
        available = [
            (p, m) for p, m, env in entries if all(os.environ.get(v) for v in env)
        ]
        if not available:
            continue

        ran_anything = True
        print(f"\n{'=' * 100}")
        print(f"{label} — {len(available)} host(s) with credentials")
        print("=" * 100)

        print("\n  Configured rates (input/cached/output per 1M tokens):")
        for prov, model in available:
            print(f"    {prov:<12} {_rate_summary(prov, model):>26}  {model}")

        print(f"\n  Live call — prompt: {PROMPT!r}")
        results = []
        for prov, model in available:
            result = await run_host(prov, model, use_gateway=use_gateway)
            if result is not None:
                results.append(result)

        _print_ranking(label, results)

        if results:
            sample = results[0]
            print(f"\n    Sample answer ({sample['provider']}):")
            print(textwrap.fill(sample["content"].strip(), width=94,
                                initial_indent="      ", subsequent_indent="      "))

    if not ran_anything:
        print("\nNo usable credentials found. Set one or more of:")
        print("  FIREWORKS_API_KEY | TOGETHER_API_KEY | BASETEN_API_KEY")
        print("  DEEPINFRA_API_KEY | NOVITA_API_KEY | NEBIUS_API_KEY | MOONSHOT_API_KEY")
        return

    print("\n" + "=" * 100)
    print("Done!")


if __name__ == "__main__":
    run_demo(
        main,
        description=__doc__,
        providers={p for _, hosts in MODEL_FAMILIES for p, _, _ in hosts},
    )
