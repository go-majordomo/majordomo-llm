#!/usr/bin/env python3
"""
Find out which Claude models Bedrock Mantle serves that ``llm_config.yaml`` lacks.

Usage:
    uv run python scripts/check_bedrock_mantle_models.py
    uv run python scripts/check_bedrock_mantle_models.py --model anthropic.claude-opus-5-5
    uv run python scripts/check_bedrock_mantle_models.py --all

AWS lists a Claude model on Bedrock Mantle some time after Anthropic ships it,
and the docs lag the endpoint in both directions. This script asks the endpoint
itself. By default the candidates are every model under ``anthropic`` in
``llm_config.yaml`` that has no ``bedrock_mantle`` counterpart, translated to the
Mantle ID format (``anthropic.`` prefix, no date suffix). ``--all`` also
re-checks the models already registered, and ``--model`` probes explicit IDs.

Each candidate gets up to two real requests:

  1. A minimal message. A 404 means the model is not served in this region.
  2. If it is served, the same message with a forced ``tool_choice``. A 400
     means the entry needs ``supports_forced_tool_choice: false``.

Unlike ``check_max_tokens.py`` these requests run inference, so the sweep is
not free: each one is capped at a few dozen output tokens, a fraction of a
cent per model.

For every model that is served, a ``bedrock_mantle`` YAML entry is printed with
the capability flags the probe established. Its rates are copied from the
first-party ``anthropic`` entry as a starting point only: Bedrock is billed by
AWS, so confirm them at https://aws.amazon.com/bedrock/pricing/ before pasting.

Credentials are read from the environment or a local ``.env``:
``AWS_BEARER_TOKEN_BEDROCK`` and ``AWS_REGION`` (or ``AWS_DEFAULT_REGION``).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from majordomo_llm.providers.bedrock_mantle import BedrockMantle  # noqa: E402

load_dotenv()

CONFIG_PATH = Path(__file__).parent.parent / "src" / "majordomo_llm" / "llm_config.yaml"

MANTLE_PREFIX = "anthropic."

#: First-party snapshot IDs end in a release date that Mantle IDs drop.
DATE_SUFFIX = re.compile(r"-\d{8}$")

#: Enough for the model to answer "hi"; the probe only needs the status code.
PROBE_MAX_TOKENS = 32

#: A probe answers in a second or two. The SDK default is ten minutes with two
#: retries, which turns one unreachable model into a half-hour hang.
PROBE_TIMEOUT_SECONDS = 30.0

PROBE_TOOL: dict[str, Any] = {
    "name": "Answer",
    "description": "Provide the answer.",
    "input_schema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
}

#: Keys copied from the first-party entry into the suggested Mantle entry.
#: Rates are a starting point; the capability flags describe the model itself.
COPIED_KEYS = (
    "input_cost",
    "output_cost",
    "cached_input_cost",
    "cache_write_cost",
    "supports_temperature_top_p",
    "supports_structured_outputs",
)

DIVIDER = "═" * 72


@dataclass
class Result:
    model: str
    served: bool | None
    forced_tool_choice: bool | None
    detail: str

    @property
    def status(self) -> str:
        if self.served is None:
            return "UNKNOWN"
        return "SERVED" if self.served else "NOT SERVED"


def to_mantle_id(anthropic_model: str) -> str:
    """Translate a first-party model ID to the Bedrock Mantle format."""
    return MANTLE_PREFIX + DATE_SUFFIX.sub("", anthropic_model)


def load_candidates(config: dict[str, Any], include_registered: bool) -> list[str]:
    """Mantle IDs for the first-party Claude models, minus effort/thinking profiles."""
    registered = set(config["bedrock_mantle"]["models"])
    candidates: list[str] = []
    for model, attrs in config["anthropic"]["models"].items():
        # A profile entry re-registers an existing SKU under another key.
        if "model" in attrs:
            continue
        mantle_id = to_mantle_id(model)
        if mantle_id in candidates:
            continue
        if include_registered or mantle_id not in registered:
            candidates.append(mantle_id)
    return candidates


def first_party_attrs(config: dict[str, Any], mantle_id: str) -> dict[str, Any] | None:
    """The ``anthropic`` config entry a Mantle ID corresponds to, if any."""
    for model, attrs in config["anthropic"]["models"].items():
        if "model" not in attrs and to_mantle_id(model) == mantle_id:
            return dict(attrs)
    return None


def _error_message(error: anthropic.APIStatusError) -> str:
    return " ".join(str(error.message).split())[:200]


async def probe(model: str) -> Result:
    """Establish whether Mantle serves ``model`` and whether it allows forced tools."""
    llm = BedrockMantle(model=model, input_cost=0.0, output_cost=0.0)
    client = llm.client.with_options(timeout=PROBE_TIMEOUT_SECONDS, max_retries=0)
    messages: list[Any] = [{"role": "user", "content": "hi"}]

    try:
        await client.messages.create(
            model=model, max_tokens=PROBE_MAX_TOKENS, messages=messages
        )
    except anthropic.NotFoundError as e:
        return Result(model, False, None, _error_message(e))
    except anthropic.APIStatusError as e:
        return Result(model, None, None, f"HTTP {e.status_code}: {_error_message(e)}")
    except anthropic.APIConnectionError as e:
        return Result(model, None, None, f"connection error: {e}")

    try:
        await client.messages.create(
            model=model,
            max_tokens=PROBE_MAX_TOKENS,
            messages=messages,
            tools=[PROBE_TOOL],
            tool_choice={"type": "tool", "name": PROBE_TOOL["name"]},
        )
    except anthropic.BadRequestError as e:
        message = _error_message(e)
        if "tool_choice" in message:
            return Result(model, True, False, message)
        return Result(model, True, None, f"forced tool_choice probe: HTTP 400: {message}")
    except anthropic.APIStatusError as e:
        return Result(
            model,
            True,
            None,
            f"forced tool_choice probe: HTTP {e.status_code}: {_error_message(e)}",
        )
    except anthropic.APIConnectionError as e:
        return Result(model, True, None, f"forced tool_choice probe: connection error: {e}")

    return Result(model, True, True, "")


def suggested_entry(result: Result, attrs: dict[str, Any] | None) -> str:
    """Render a ``bedrock_mantle`` YAML entry for a served model."""
    lines = [f"    {result.model}:"]
    if attrs is None:
        lines.append("      # No first-party entry to copy from: fill in rates and flags.")
    else:
        for key in COPIED_KEYS:
            if key in attrs:
                value = attrs[key]
                rendered = str(value).lower() if isinstance(value, bool) else str(value)
                lines.append(f"      {key}: {rendered}")
    if result.forced_tool_choice is False:
        lines.append("      supports_forced_tool_choice: false")
    elif result.forced_tool_choice is None:
        lines.append("      # supports_forced_tool_choice: could not be established, see above")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="ID",
        help="probe this Mantle model ID instead of the derived candidates (repeatable)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="also re-check models already registered under bedrock_mantle",
    )
    args = parser.parse_args()

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    missing = [
        name
        for name, present in (
            ("AWS_BEARER_TOKEN_BEDROCK", bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))),
            ("AWS_REGION", bool(region)),
        )
        if not present
    ]
    if missing:
        print(
            f"Missing {', '.join(missing)}. Set them in the environment or a local "
            ".env, then re-run.",
            file=sys.stderr,
        )
        return 2

    config = yaml.safe_load(CONFIG_PATH.read_text())
    candidates = args.model or load_candidates(config, include_registered=args.all)
    if not candidates:
        print("Every first-party Claude model already has a bedrock_mantle entry.")
        return 0

    print(f"\n{DIVIDER}")
    print(f"Probing {len(candidates)} model IDs on Bedrock Mantle ({region})")
    print(DIVIDER)

    results = [await probe(model) for model in candidates]

    for r in results:
        marker = {"SERVED": "✓", "NOT SERVED": "·", "UNKNOWN": "?"}[r.status]
        line = f"{marker} {r.model}  {r.status.lower()}"
        if r.served:
            forced = {True: "accepted", False: "rejected", None: "unknown"}[r.forced_tool_choice]
            line += f", forced tool_choice {forced}"
        print(line)
        if r.detail and r.status != "NOT SERVED":
            print(f"    {r.detail}")

    registered = set(config["bedrock_mantle"]["models"])
    to_add = [r for r in results if r.served and r.model not in registered]
    gone = [r for r in results if r.served is False and r.model in registered]
    unknown = [
        r for r in results if r.served is None or (r.served and r.forced_tool_choice is None)
    ]

    if to_add:
        print(f"\n{DIVIDER}")
        print("Served but not registered. Paste under bedrock_mantle.models after")
        print("confirming the rates at https://aws.amazon.com/bedrock/pricing/ :")
        print(DIVIDER)
        for r in to_add:
            print(suggested_entry(r, first_party_attrs(config, r.model)))

    if gone:
        print(f"\nRegistered but no longer served in {region}:")
        for r in gone:
            print(f"  {r.model}")

    print(f"\n{len(to_add)} to add, {len(gone)} registered but not served, "
          f"{len(unknown)} unresolved\n")
    return 1 if gone or unknown else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
