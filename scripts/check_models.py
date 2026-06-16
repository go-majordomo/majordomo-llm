#!/usr/bin/env python3
"""
Check llm_config.yaml against LiteLLM's public model pricing database.

Usage:
    uv run python scripts/check_models.py

Fetches https://github.com/BerriAI/litellm model_prices_and_context_window.json
(no API keys required) and compares it against llm_config.yaml, reporting:

  NEW     — in LiteLLM but absent from config (YAML snippet with real costs printed)
  REMOVED — in config but absent from LiteLLM (checked against deprecated_models)
"""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "src" / "majordomo_llm" / "llm_config.yaml"
LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
TOKENS_PER_MILLION = 1_000_000

# LiteLLM litellm_provider values → our provider keys
PROVIDER_MAP: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "deepseek": "deepseek",
    "cohere_chat": "cohere",
    "cohere": "cohere",
}

DIVIDER = "═" * 64


# ── Config ────────────────────────────────────────────────────────────────────


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── LiteLLM fetch ─────────────────────────────────────────────────────────────


@dataclass
class ModelInfo:
    model_id: str
    input_cost: float  # USD per million tokens
    output_cost: float  # USD per million tokens


def fetch_litellm_models() -> dict[str, dict[str, ModelInfo]]:
    """Return {our_provider: {model_id: ModelInfo}} from LiteLLM's public DB."""
    print(f"Fetching {LITELLM_URL} ...")
    with urllib.request.urlopen(LITELLM_URL, timeout=15) as resp:  # noqa: S310
        data: dict = json.loads(resp.read())

    result: dict[str, dict[str, ModelInfo]] = {}

    for key, info in data.items():
        if not isinstance(info, dict):
            continue

        our_provider = PROVIDER_MAP.get(info.get("litellm_provider", ""))
        if not our_provider:
            continue

        if info.get("mode") not in ("chat", "completion"):
            continue

        # Strip provider prefix from key (e.g. "gemini/gemini-2.5-pro" → "gemini-2.5-pro")
        model_id = key.split("/", 1)[-1] if "/" in key else key

        input_per_token = info.get("input_cost_per_token")
        output_per_token = info.get("output_cost_per_token")
        if input_per_token is None or output_per_token is None:
            continue

        result.setdefault(our_provider, {})[model_id] = ModelInfo(
            model_id=model_id,
            input_cost=round(input_per_token * TOKENS_PER_MILLION, 4),
            output_cost=round(output_per_token * TOKENS_PER_MILLION, 4),
        )

    return result


# ── Diff ──────────────────────────────────────────────────────────────────────


@dataclass
class ProviderDiff:
    provider: str
    new_models: list[ModelInfo] = field(default_factory=list)
    deprecated_ok: list[str] = field(default_factory=list)
    deprecated_missing: list[str] = field(default_factory=list)


def compute_diff(
    config: dict,
    provider: str,
    litellm_models: dict[str, ModelInfo],
) -> ProviderDiff:
    diff = ProviderDiff(provider=provider)
    config_models = set((config.get(provider) or {}).get("models", {}).keys())
    deprecated = set((config.get("deprecated_models") or {}).get(provider, {}).keys())
    litellm_ids = set(litellm_models.keys())

    diff.new_models = sorted(
        [
            m
            for mid, m in litellm_models.items()
            if mid not in config_models and mid not in deprecated
        ],
        key=lambda m: m.model_id,
    )

    for m in sorted(config_models - litellm_ids):
        if m in deprecated:
            diff.deprecated_ok.append(m)
        else:
            diff.deprecated_missing.append(m)

    return diff


# ── Reporting ─────────────────────────────────────────────────────────────────


def print_new_models(diffs: dict[str, ProviderDiff]) -> None:
    any_new = any(d.new_models for d in diffs.values())

    print(f"\n{DIVIDER}")
    print("NEW MODELS  (in LiteLLM DB — not in config)")
    print(DIVIDER)

    if not any_new:
        print("  (none)")
        return

    for provider, diff in diffs.items():
        if not diff.new_models:
            continue
        print(f"\n[{provider}]")
        for m in diff.new_models:
            print(f"  + {m.model_id}  (input={m.input_cost}, output={m.output_cost} $/M)")

    print(f"\n{'─' * 64}")
    print("Paste into llm_config.yaml under the appropriate provider:")
    print(f"{'─' * 64}")
    for provider, diff in diffs.items():
        if not diff.new_models:
            continue
        print(f"\n# {provider}")
        for m in diff.new_models:
            print(f"    {m.model_id}:")
            print(f"      input_cost: {m.input_cost}")
            print(f"      output_cost: {m.output_cost}")
            print(f"      # supports_temperature_top_p: false  # uncomment if needed")


def print_removed_models(config: dict, diffs: dict[str, ProviderDiff]) -> None:
    any_removed = any(d.deprecated_ok or d.deprecated_missing for d in diffs.values())

    print(f"\n{DIVIDER}")
    print("REMOVED MODELS  (in config — absent from LiteLLM DB)")
    print(DIVIDER)

    if not any_removed:
        print("  (none)")
        return

    deprecated_cfg = config.get("deprecated_models") or {}

    for provider, diff in diffs.items():
        for m in diff.deprecated_ok:
            replacement = deprecated_cfg.get(provider, {}).get(m, "?")
            print(f"  [{provider}] {m}")
            print(f"    ✓ already in deprecated_models → {replacement}")

        for m in diff.deprecated_missing:
            print(f"  [{provider}] {m}")
            print(f"    ✗ NOT in deprecated_models — needs a mapping!")

    unhandled = [(p, m) for p, d in diffs.items() for m in d.deprecated_missing]
    if unhandled:
        print(f"\n{'─' * 64}")
        print("Add to deprecated_models in llm_config.yaml:")
        print(f"{'─' * 64}")
        print("\ndeprecated_models:")
        by_provider: dict[str, list[str]] = {}
        for provider, model_id in unhandled:
            by_provider.setdefault(provider, []).append(model_id)
        for provider, models in by_provider.items():
            print(f"  {provider}:")
            for m in models:
                print(f"    {m}: REPLACEMENT_MODEL_ID")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    config = load_config()

    try:
        litellm = fetch_litellm_models()
    except Exception as e:
        print(f"Failed to fetch LiteLLM model DB: {e}", file=sys.stderr)
        sys.exit(1)

    provider_counts = {p: len(models) for p, models in litellm.items()}
    print(f"  fetched — {provider_counts}")

    diffs = {
        provider: compute_diff(config, provider, litellm.get(provider, {}))
        for provider in config
        if provider not in ("deprecated_models", "aliases")
    }

    print_new_models(diffs)
    print_removed_models(config, diffs)

    any_issues = any(d.new_models or d.deprecated_missing for d in diffs.values())
    print(f"\n{DIVIDER}")
    print("Action required — see above." if any_issues else "✓ Config is up to date.")
    print()


if __name__ == "__main__":
    main()
