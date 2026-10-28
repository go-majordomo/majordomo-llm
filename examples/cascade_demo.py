#!/usr/bin/env python3
"""Demo script showcasing LLMCascade failover and the alias registry.

An LLMCascade wraps an ordered list of (provider, model) pairs and tries each in
turn, catching ProviderError and falling through to the next. It is CLIENT-side
failover: the decision lives in your process, and every hop is a real attempt.
(Contrast with the `majordomo` provider in routing_demo.py, which pushes the
choice server-side to the gateway.)

Aliases give those chains a name. llm_config.yaml ships several — `glm-5.3`,
`kimi-k3`, `inkling` and friends — so `get_llm_by_alias("glm-5.3")`
returns a ready-made cascade rather than a single model.

This script demonstrates:

- Listing the alias registry, single-model and cascade alike
- Running a cascade alias end to end and reporting which member answered,
  read from the response: response.provider / response.model name it, and
  response.attempted lists every member tried, ending with it
- Forced failover: a deliberately broken first hop, proving the second is used
- register_alias() for a chain assembled at runtime

Prerequisites:
    1. Install dependencies: uv sync --all-extras
    2. Set at least two of the open-weight keys so failover has somewhere to go:
       - FIREWORKS_API_KEY, TOGETHER_API_KEY, BASETEN_API_KEY

Gateway mode (--gateway) routes through Majordomo Steward:
    - MAJORDOMO_GATEWAY_URL (defaults to http://localhost:7680)
    - MAJORDOMO_API_KEY (required)
"""

import os
import textwrap

from shared import gateway_kwargs, run_demo

from majordomo_llm import (
    LLMCascade,
    LLMResponse,
    get_aliases,
    get_llm_by_alias,
    register_alias,
)

PROMPT = "Name the three primary additive colors. Answer in one short sentence."

def report_answer(response: LLMResponse) -> None:
    """Print which member answered and, after a fallback, every member tried."""
    print(f"  Answered by: {response.provider}/{response.model}")
    if len(response.attempted) > 1:
        tried = " -> ".join(f"{p}/{m}" for p, m in response.attempted)
        print(f"  Tried:       {tried}")


# Candidate hosts for a runtime-assembled GLM-5.3 cascade. All three charge the
# same input and output rate, so Baseten leads on its cheaper cached reads.
# Whichever have keys are used, in this order.
GLM_HOSTS: list[tuple[str, str, str]] = [
    ("baseten", "zai-org/GLM-5.3", "BASETEN_API_KEY"),
    ("fireworks", "accounts/fireworks/models/glm-5p3", "FIREWORKS_API_KEY"),
    ("together", "zai-org/GLM-5.3", "TOGETHER_API_KEY"),
]


def demo_registry() -> None:
    """Print the alias registry, separating single models from cascades."""
    print("\n" + "-" * 90)
    print("Demo 1: The alias registry")
    print("-" * 90)

    aliases = get_aliases()
    singles = {k: v for k, v in aliases.items() if not isinstance(v, list)}
    cascades = {k: v for k, v in aliases.items() if isinstance(v, list)}

    print(f"\n  Single-model aliases ({len(singles)}):")
    for name, (provider, model) in sorted(singles.items()):
        print(f"    {name:<28} -> {provider}/{model}")

    print(f"\n  Cascade aliases ({len(cascades)}):")
    for name, chain in sorted(cascades.items()):
        hops = " -> ".join(f"{p}/{m}" for p, m in chain)
        print(f"    {name:<28} -> {hops}")


async def demo_cascade_alias(use_gateway: bool = False) -> bool:
    """Resolve a cascade alias and run it, reporting which member answered."""
    print("\n" + "-" * 90)
    print("Demo 2: Running a cascade alias")
    print("-" * 90)

    alias = "glm-5.3"
    llm = get_llm_by_alias(alias, **gateway_kwargs(use_gateway, feature="cascade-demo"))
    chain = " -> ".join(f"{m.provider}/{m.model}" for m in llm.llms)
    print(f"\n  get_llm_by_alias({alias!r}) -> {chain}")

    try:
        response = await llm.get_response(PROMPT)
    except Exception as e:
        print(f"  Every member failed: {e}")
        return False

    report_answer(response)
    print("  Content:")
    print(textwrap.fill(response.content.strip(), width=88,
                        initial_indent="    ", subsequent_indent="    "))
    print(f"  Tokens: {response.input_tokens} in / {response.output_tokens} out "
          f"| Cost: ${response.total_cost:.6f}")
    return True


async def demo_forced_failover(hosts: list[tuple[str, str]]) -> bool:
    """Break the first hop on purpose so the fallback demonstrably takes over."""
    print("\n" + "-" * 90)
    print("Demo 3: Forced failover")
    print("-" * 90)

    if len(hosts) < 2:
        print("\n  Skipped — needs two hosts with keys to show a fallback.")
        return True

    primary, fallback = hosts[0], hosts[1]
    print(f"\n  Primary  : {primary[0]}/{primary[1]}  (given an invalid API key)")
    print(f"  Fallback : {fallback[0]}/{fallback[1]}")

    # A single api_key applies to every member, so build the cascade with a
    # deliberately bad key for the primary and let the fallback pick its own key
    # up from the environment.
    broken = LLMCascade([primary], api_key="sk-invalid-on-purpose")
    healthy = LLMCascade([fallback])
    cascade = LLMCascade([primary, fallback])
    cascade.llms = [broken.llms[0], healthy.llms[0]]

    try:
        response = await cascade.get_response(PROMPT)
    except Exception as e:
        print(f"  Both members failed: {e}")
        return False

    report_answer(response)
    if (response.provider, response.model) == fallback and len(response.attempted) == 2:
        print("  Failover worked — the broken primary was skipped.")
    else:
        print("  Unexpected: the primary answered despite the invalid key.")
    return True


async def demo_runtime_alias(hosts: list[tuple[str, str]]) -> bool:
    """Register a cascade at runtime and resolve it by name."""
    print("\n" + "-" * 90)
    print("Demo 4: Registering a cascade at runtime")
    print("-" * 90)

    if len(hosts) < 2:
        print("\n  Skipped — register_alias requires at least 2 members for a cascade.")
        return True

    name = "glm-cheapest-first"
    register_alias(name, hosts)
    hops = " -> ".join(f"{p}/{m}" for p, m in hosts)
    print(f"\n  register_alias({name!r}, [...]) -> {hops}")

    llm = get_llm_by_alias(name)
    try:
        response = await llm.get_response(PROMPT)
    except Exception as e:
        print(f"  Every member failed: {e}")
        return False

    report_answer(response)
    print(f"  Cost: ${response.total_cost:.6f}")
    return True


async def main(use_gateway: bool = False, provider: str | None = None) -> None:
    # `provider` is always None here: this demo opts out of the --provider flag
    # (run_demo(..., provider_filter=False)) because a cascade is defined by its
    # chain, not by a single provider. The parameter stays for run_demo's uniform
    # main(use_gateway, provider) signature.
    del provider
    print("=" * 90)
    print("majordomo-llm: Cascade & Alias Demo")
    print("=" * 90)

    if use_gateway:
        print("Routing through the Majordomo gateway (Steward).")
    demo_registry()

    hosts = [
        (p, m) for p, m, env in GLM_HOSTS if os.environ.get(env)
    ]
    if not hosts:
        print("\nNo open-weight credentials found — live demos skipped. Set one of:")
        print("  BASETEN_API_KEY | FIREWORKS_API_KEY | TOGETHER_API_KEY")
        return

    print(f"\nHosts with credentials: {', '.join(p for p, _ in hosts)}")

    results = [
        await demo_cascade_alias(use_gateway=use_gateway),
        await demo_forced_failover(hosts),
        await demo_runtime_alias(hosts),
    ]

    print("\n" + "=" * 90)
    print(f"Done! {sum(results)}/{len(results)} demos succeeded.")


if __name__ == "__main__":
    run_demo(main, description=__doc__, provider_filter=False)
