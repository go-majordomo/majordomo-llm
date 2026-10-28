#!/usr/bin/env python3
"""
Find how each Anthropic model spells "thinking off", live.

Usage:
    uv run python scripts/check_anthropic_thinking.py
    uv run python scripts/check_anthropic_thinking.py --model claude-opus-5-5

Models spell "off" differently. Older models take ``thinking: {"type":
"disabled"}``. ``claude-sonnet-5-5`` rejects that with a 400 and asks for
``{"type": "between_tools"}``. Fable 5 thinks on every request and accepts
neither. ``llm_config.yaml`` records each model's spelling as ``thinking_off``,
and ``get_llm_instance(thinking="disabled")`` sends it. This script is how
those entries are established, and how to re-check them when a model changes.

For every Anthropic model registered in the config (each upstream model once,
however many profile entries share it) it sends one trivial classification
prompt under each setting below. It prints the HTTP status, output tokens
(Anthropic bills thinking as output and counts it there), whether a thinking
block came back, and the answer or the error. It then prints the
``thinking_off`` value the results support next to the configured one.

The requests go straight to the Messages API with the anthropic SDK rather than
through ``get_llm_instance``, so a value the provider would reject can still be
measured.

This runs inference: four small calls per model, each capped at
``PROBE_MAX_TOKENS`` output tokens. A full sweep costs a few cents. One sample
per setting is noisy, so treat a small difference in output tokens as no
difference. The status code and whether a thinking block came back are the
reliable signals.

Credentials are read from the environment or a local ``.env``:
``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent.parent / "src" / "majordomo_llm" / "llm_config.yaml"

#: Enough room for a short burst of thinking on a one-word answer.
PROBE_MAX_TOKENS = 2048

PROBE_TIMEOUT_SECONDS = 120.0

#: Label -> ``thinking`` field sent, or ``None`` to send none. ``adaptive`` is
#: the "on" reference.
SETTINGS: tuple[tuple[str, dict[str, Any] | None], ...] = (
    ("nothing sent", None),
    ("disabled", {"type": "disabled"}),
    ("between_tools", {"type": "between_tools"}),
    ("adaptive", {"type": "adaptive"}),
)

SYSTEM = "Classify the support ticket as one word: billing, technical, or account."
USER = "Why was I charged $49.99 when the website showed $39.99?"

DIVIDER = "═" * 92


@dataclass
class Result:
    label: str
    status: int | None
    output_tokens: int | None = None
    thought: bool = False
    seconds: float | None = None
    detail: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == 200


def probe(client: anthropic.Anthropic, model: str, label: str, thinking: Any) -> Result:
    """Send one request under one setting and summarize what came back."""
    kwargs: dict[str, Any] = {} if thinking is None else {"thinking": thinking}
    start = time.perf_counter()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=PROBE_MAX_TOKENS,
            system=SYSTEM,
            messages=[{"role": "user", "content": USER}],
            **kwargs,
        )
    except anthropic.APIStatusError as e:
        return Result(
            label, e.status_code, seconds=time.perf_counter() - start, detail=_error_message(e)
        )
    except anthropic.APIConnectionError as e:
        return Result(label, None, detail=f"connection error: {e}")

    answer = " ".join(
        "".join(block.text for block in response.content if block.type == "text").split()
    )
    return Result(
        label,
        200,
        output_tokens=response.usage.output_tokens,
        thought=any(block.type in ("thinking", "redacted_thinking") for block in response.content),
        seconds=time.perf_counter() - start,
        detail=answer,
    )


def _error_message(error: anthropic.APIStatusError) -> str:
    """The API's own explanation of a rejection, without the SDK's wrapping.

    It often names the value the model wants instead, so it is printed in full.
    """
    body = error.body
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict) and isinstance(inner.get("message"), str):
            return " ".join(inner["message"].split())
    return " ".join(str(error.message).split())


def supported_thinking_off(results: dict[str, Result]) -> str:
    """The ``thinking_off`` value these results support.

    ``disabled`` when the model accepts it (also the value used when the key is
    absent), else ``between_tools`` when that is accepted, else ``unsupported``.
    """
    if results["disabled"].accepted:
        return "disabled"
    if results["between_tools"].accepted:
        return "between_tools"
    return "unsupported"


def format_row(result: Result) -> str:
    status = "-" if result.status is None else str(result.status)
    out = "-" if result.output_tokens is None else str(result.output_tokens)
    thought = "yes" if result.thought else ("-" if not result.accepted else "no")
    secs = "-" if result.seconds is None else f"{result.seconds:.1f}"
    return f"  {result.label:<14} {status:>4} {out:>6} {thought:>8} {secs:>6}  {result.detail[:52]}"


def check_model(client: anthropic.Anthropic, model: str, configured: str) -> bool:
    """Probe one model. Returns whether the configured value matches the results."""
    print(f"\nanthropic / {model}")
    print(f"  {'thinking':<14} {'http':>4} {'out':>6} {'thought':>8} {'secs':>6}  answer / error")
    results: dict[str, Result] = {}
    for label, thinking in SETTINGS:
        results[label] = probe(client, model, label, thinking)
        result = results[label]
        print(format_row(result), flush=True)
        if not result.accepted and len(result.detail) > 52:
            print(f"      {result.detail}")

    supported = supported_thinking_off(results)
    matches = supported == configured
    verdict = "matches" if matches else "DIFFERS"
    print(f"  thinking_off: configured {configured}, results support {supported} ({verdict})")
    off = results.get(supported)
    if off is not None and off.thought:
        print(f"  note: '{supported}' was accepted but a thinking block still came back")
    return matches


def configured_models(config: dict[str, Any]) -> dict[str, str]:
    """Upstream model id -> its configured ``thinking_off`` (``disabled`` if absent).

    Profile entries share an upstream model through the ``model`` key; it is
    probed once, under the first entry's value.
    """
    models: dict[str, str] = {}
    for key, attrs in config["anthropic"]["models"].items():
        upstream = attrs.get("model", key)
        models.setdefault(upstream, attrs.get("thinking_off", "disabled"))
    return models


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="ID",
        help="probe only this model ID (repeatable); must be registered under anthropic",
    )
    args = parser.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC_API_KEY is not set in the environment or a local .env", file=sys.stderr)
        return 2

    models = configured_models(yaml.safe_load(CONFIG_PATH.read_text()))
    wanted = {m: v for m, v in models.items() if not args.model or m in args.model}
    if not wanted:
        print("No registered Anthropic model matches --model.", file=sys.stderr)
        return 2

    client = anthropic.Anthropic(api_key=key, timeout=PROBE_TIMEOUT_SECONDS, max_retries=0)
    print(DIVIDER)
    print(f"anthropic: {len(wanted)} model(s), {len(SETTINGS)} settings each")
    print(DIVIDER)
    differing = [m for m, configured in wanted.items() if not check_model(client, m, configured)]

    print()
    if differing:
        print(
            "thinking_off in llm_config.yaml differs from the results for: " + ", ".join(differing)
        )
        return 1
    print("thinking_off in llm_config.yaml matches the results for every model probed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
