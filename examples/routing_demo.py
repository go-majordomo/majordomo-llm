#!/usr/bin/env python3
"""Demo script showcasing the `majordomo` provider's server-side optimal routing.

Every other provider names a concrete backend. The `majordomo` provider names a
canonical open-weight model — `glm-5.2`, `kimi-k3`, `deepseek-v4-pro` and so on —
and lets Majordomo Steward choose which host serves it, per request.

Two things follow from that, and this demo shows both:

- The backend is only known AFTER the call. It comes back on the response as
  `routed_provider` / `routed_model`, populated from the gateway's
  X-Majordomo-Routed-Provider / X-Majordomo-Routed-Model headers.
- Cost cannot come from a fixed config entry. The usage is priced against the
  routed pair's rates in llm_config.yaml, so the same token counts cost
  different amounts depending on where the gateway sent them — GLM-5.2's cached
  read is $0.14/M on Baseten and DeepInfra but $0.26/M on Novita.

Contrast with cascade_demo.py: an LLMCascade is CLIENT-side failover, where your
process tries hosts in order and every hop is a real attempt. This is
SERVER-side selection — one request, the gateway decides. The two compose.

Not to be confused with the --gateway flag on the other demos. Two different
features that happen to share MAJORDOMO_API_KEY:

    --gateway on other demos   You name a concrete provider and model; Steward
    (usage tracking)           fronts it to record spend and attribute cost. The
                               model that runs is exactly the one you asked for.

    the majordomo provider     You name a canonical model; Steward decides which
    (this demo, routing)       backend runs it. You find out afterwards, from
                               routed_provider / routed_model.

No other demo touches the majordomo provider, and the shared provider sweep in
shared.py explicitly refuses to select it. Running THIS script is the opt-in.

Prerequisites:
    1. Install dependencies: uv sync --all-extras
    2. A running Majordomo Steward instance with backend provider keys configured
    3. Set:
       - MAJORDOMO_API_KEY (required — sent as X-Majordomo-Key)
       - MAJORDOMO_GATEWAY_URL (defaults to http://localhost:7680)

    Unlike the other demos, this one CANNOT run without a gateway: the provider
    raises ConfigurationError without a base_url, by design.

Usage:
    uv run python examples/routing_demo.py
    uv run python examples/routing_demo.py --model glm-5.2 --model kimi-k3
    uv run python examples/routing_demo.py --gateway-url http://steward.internal:7680

    The interesting flag is --model, which takes a CANONICAL model name: choosing
    the provider is the gateway's job, which is the whole point. `--provider
    majordomo` is accepted but redundant, and is rejected by every other demo.
"""

import argparse
import asyncio
import os
import textwrap

from shared import DEFAULT_GATEWAY_URL
from tenacity import RetryError

from majordomo_llm import get_llm_instance, get_model_pricing, get_supported_models

PROMPT = "In one sentence, what problem does an inference gateway solve?"

# Canonical names the gateway resolves. These are NOT host model IDs — the whole
# point is that the caller does not pick a host.
CANONICAL_MODELS = ["glm-5.2", "kimi-k3", "deepseek-v4-pro"]


def explain(exc: Exception) -> str:
    """Render an error legibly.

    A connection failure is retryable, so tenacity re-raises it wrapped as
    ``RetryError[<Future ...>]``, which tells the reader nothing. Unwrap to the
    underlying cause — usually "Steward is not running".
    """
    if isinstance(exc, RetryError):
        cause = exc.last_attempt.exception()
        if cause is not None:
            exc = cause
    text = str(exc)
    if "connect" in text.lower():
        return f"{text}  <- is Steward running and reachable at the URL above?"
    return text


def describe_route(response: object) -> str:
    """Render where the gateway actually sent the request."""
    if response.routed_provider is None:
        return (
            "gateway did not report a route "
            "(no X-Majordomo-Routed-Provider header on the response)"
        )
    route = f"{response.routed_provider}/{response.routed_model}"
    pricing = get_model_pricing(response.routed_provider, response.routed_model or "")
    if pricing is None:
        return f"{route}  (routed pair not in llm_config.yaml — cost degrades to $0.00)"
    cached = (
        f"{pricing.cached_input_cost:g}"
        if pricing.cached_input_cost is not None
        else "-"
    )
    return (
        f"{route}  (rates ${pricing.input_cost:g}/{cached}/{pricing.output_cost:g} "
        "per 1M in/cached/out)"
    )


async def demo_route(model: str, base_url: str) -> bool:
    """Send one canonical-model request and report the route the gateway chose."""
    print(f"\n  [majordomo/{model}]")
    try:
        llm = get_llm_instance("majordomo", model, base_url=base_url)
        response = await llm.get_response(PROMPT, system_prompt="Be concise.")
    except Exception as e:
        print(f"    Error: {explain(e)}")
        return False

    print(f"    Routed to : {describe_route(response)}")
    print(f"    Tokens    : {response.input_tokens} in / {response.output_tokens} out "
          f"| cached {response.cached_tokens}")
    print(f"    Cost      : ${response.total_cost:.6f} "
          f"(input ${response.input_cost:.6f} + output ${response.output_cost:.6f})")
    print("    Content   :")
    print(textwrap.fill(response.content.strip(), width=88,
                        initial_indent="      ", subsequent_indent="      "))
    return True


async def demo_streaming_route(model: str, base_url: str) -> bool:
    """Streaming carries the same routing metadata, resolved at finalization."""
    print(f"\n  [majordomo/{model}] streaming")
    try:
        llm = get_llm_instance("majordomo", model, base_url=base_url)
        stream = await llm.get_response_stream(PROMPT, system_prompt="Be concise.")
        chunks = 0
        async for _ in stream:
            chunks += 1
        response = await stream.collect()
    except Exception as e:
        print(f"    Error: {explain(e)}")
        return False

    print(f"    Routed to : {describe_route(response)}")
    print(f"    Chunks    : {chunks}")
    print(f"    Tokens    : {response.input_tokens} in / {response.output_tokens} out")
    print(f"    Cost      : ${response.total_cost:.6f} "
          "(priced against the routed backend, not the gateway entry)")
    print("    Content   :")
    print(textwrap.fill(response.content.strip(), width=88,
                        initial_indent="      ", subsequent_indent="      "))
    return True


async def main(models: list[str], base_url: str) -> None:
    print("=" * 90)
    print("majordomo-llm: Optimal Routing Demo (majordomo gateway provider)")
    print("=" * 90)

    print(f"\nGateway: {base_url}")
    print(f"Canonical models the gateway can route: "
          f"{', '.join(get_supported_models('majordomo'))}")

    print("\n" + "-" * 90)
    print("Demo 1: Where does the gateway send each canonical model?")
    print("-" * 90)
    results = [await demo_route(model, base_url) for model in models]

    print("\n" + "-" * 90)
    print("Demo 2: Streaming through the router")
    print("-" * 90)
    results.append(await demo_streaming_route(models[0], base_url))

    print("\n" + "=" * 90)
    print(f"Done! {sum(results)}/{len(results)} calls succeeded.")


def cli() -> None:
    """Parse this demo's own flags — it does not use the shared run_demo()."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["majordomo"],
        help="Optional and redundant — this demo only exercises the majordomo "
        "provider. Accepted so that `--provider majordomo`, which every other "
        "demo rejects, works here.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="CANONICAL",
        help="Canonical model to route (repeatable). "
        f"Default: {', '.join(CANONICAL_MODELS)}.",
    )
    parser.add_argument(
        "--gateway-url",
        default=os.environ.get("MAJORDOMO_GATEWAY_URL", DEFAULT_GATEWAY_URL),
        help="Majordomo Steward URL (default: $MAJORDOMO_GATEWAY_URL "
        f"or {DEFAULT_GATEWAY_URL}).",
    )
    args = parser.parse_args()

    if not os.environ.get("MAJORDOMO_API_KEY"):
        print("MAJORDOMO_API_KEY is not set.")
        print("This demo requires a running Majordomo Steward instance — the")
        print("majordomo provider has no direct mode. Set:")
        print("  MAJORDOMO_API_KEY=...")
        print(f"  MAJORDOMO_GATEWAY_URL=...  (defaults to {DEFAULT_GATEWAY_URL})")
        return

    models = args.models or CANONICAL_MODELS
    routable = get_supported_models("majordomo")
    unknown = [m for m in models if m not in routable]
    if unknown:
        print(f"Not a canonical majordomo model: {', '.join(unknown)}")
        print(f"Choose from: {', '.join(routable)}")
        return

    asyncio.run(main(models, args.gateway_url))


if __name__ == "__main__":
    cli()
