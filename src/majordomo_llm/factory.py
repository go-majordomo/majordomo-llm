"""Factory functions for creating LLM instances."""

from __future__ import annotations

import importlib.resources
import logging
from collections.abc import Iterator
from typing import Any, cast

import yaml

from majordomo_llm.base import LLM
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers.anthropic import Anthropic
from majordomo_llm.providers.cohere import Cohere
from majordomo_llm.providers.deepseek import DeepSeek
from majordomo_llm.providers.gemini import Gemini
from majordomo_llm.providers.openai import OpenAI

logger = logging.getLogger(__name__)

#: Type for alias targets: single (provider, model) or cascade list.
AliasTarget = tuple[str, str] | list[tuple[str, str]]


def _load_llm_config() -> dict[str, dict[str, Any]]:
    """Load LLM configuration from the bundled YAML file."""
    config_file = importlib.resources.files("majordomo_llm").joinpath("llm_config.yaml")
    with config_file.open("r") as f:
        return cast(dict[str, dict[str, Any]], yaml.safe_load(f))


#: Configuration mapping for all supported providers and models.
#: Costs are specified in USD per million tokens.
LLM_CONFIG: dict[str, dict[str, Any]] = _load_llm_config()

#: Mapping of deprecated model names to their recommended replacements,
#: keyed by provider.
_DEPRECATED_MODELS: dict[str, dict[str, str]] = LLM_CONFIG.pop("deprecated_models", {})


def _validate_provider_model(provider: str, model: str, alias_name: str) -> None:
    """Validate that a provider/model pair exists in LLM_CONFIG."""
    provider_config = LLM_CONFIG.get(provider)
    if provider_config is None:
        available = ", ".join(LLM_CONFIG.keys())
        raise ConfigurationError(
            f"Alias '{alias_name}' references unknown provider '{provider}'. Available: {available}"
        )
    if model not in provider_config.get("models", {}):
        available = ", ".join(provider_config["models"].keys())
        raise ConfigurationError(
            f"Alias '{alias_name}' references unknown model '{model}' for provider "
            f"'{provider}'. Available: {available}"
        )


def _load_aliases_from_config() -> dict[str, AliasTarget]:
    """Load aliases from the 'aliases' section of LLM_CONFIG.

    Pops the aliases key from LLM_CONFIG so provider iteration stays clean.
    """
    raw_aliases = LLM_CONFIG.pop("aliases", {})
    if not raw_aliases:
        return {}

    registry: dict[str, AliasTarget] = {}
    for alias_name, alias_def in raw_aliases.items():
        if "cascade" in alias_def:
            targets: list[tuple[str, str]] = []
            for entry in alias_def["cascade"]:
                _validate_provider_model(entry["provider"], entry["model"], alias_name)
                targets.append((entry["provider"], entry["model"]))
            if len(targets) < 2:
                raise ConfigurationError(
                    f"Alias '{alias_name}' cascade must have at least 2 providers"
                )
            registry[alias_name] = targets
        elif "provider" in alias_def and "model" in alias_def:
            _validate_provider_model(alias_def["provider"], alias_def["model"], alias_name)
            registry[alias_name] = (alias_def["provider"], alias_def["model"])
        else:
            raise ConfigurationError(
                f"Alias '{alias_name}' must have either 'provider'+'model' or 'cascade' key"
            )
    return registry


_ALIAS_REGISTRY: dict[str, AliasTarget] = _load_aliases_from_config()


def get_supported_providers() -> list[str]:
    """Return a list of all supported provider names.

    Returns:
        A list of provider name strings (e.g., ["openai", "anthropic", ...]).

    Example:
        >>> providers = get_supported_providers()
        >>> "anthropic" in providers
        True
    """
    return list(LLM_CONFIG.keys())


def get_supported_models(provider: str) -> list[str]:
    """Return a list of all supported model names for the given provider.

    Args:
        provider: The LLM provider name (e.g., "openai", "anthropic").

    Returns:
        A list of model identifier strings.

    Raises:
        ConfigurationError: If the provider is not recognized.

    Example:
        >>> models = get_supported_models("anthropic")
        >>> "claude-sonnet-4-20250514" in models
        True
    """
    provider_config = LLM_CONFIG.get(provider)
    if provider_config is None:
        available = ", ".join(LLM_CONFIG.keys())
        raise ConfigurationError(f"Unknown LLM provider '{provider}'. Available: {available}")
    return list(provider_config.get("models", {}).keys())


def get_llm_instance(
    provider: str,
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    default_headers: dict[str, str] | None = None,
) -> LLM:
    """Create an LLM instance for the specified provider and model.

    This is the primary factory function for creating LLM instances. It handles
    provider-specific initialization and configuration lookup.

    Args:
        provider: The LLM provider name. One of: "openai", "anthropic", "gemini",
            "deepseek", "cohere".
        model: The model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").
        api_key: Optional API key. If not provided, the provider will fall back
            to its respective environment variable.
        base_url: Optional custom base URL for routing through a proxy.
        default_headers: Optional headers sent with every request.

    Returns:
        An LLM instance configured for the specified provider and model.

    Raises:
        ConfigurationError: If the provider or model is not recognized.

    Example:
        >>> llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")
        >>> response = await llm.get_response("Hello!")
    """
    llm_config_entry = LLM_CONFIG.get(provider)
    if llm_config_entry is None:
        available = ", ".join(LLM_CONFIG.keys())
        raise ConfigurationError(f"Unknown LLM provider '{provider}'. Available: {available}")

    llm_models = llm_config_entry["models"]
    model_attributes = llm_models.get(model)

    # Check if the requested model is deprecated and resolve to its replacement.
    deprecation_warning = None
    requested_model = None
    if model_attributes is None:
        provider_deprecated = _DEPRECATED_MODELS.get(provider, {})
        replacement = provider_deprecated.get(model)
        if replacement is not None:
            deprecation_warning = (
                f"Model '{model}' for provider '{provider}' is deprecated. "
                f"Automatically replaced with '{replacement}'."
            )
            logger.warning(deprecation_warning)
            requested_model = model
            model = replacement
            model_attributes = llm_models.get(model)

    if model_attributes is None:
        available = ", ".join(llm_models.keys())
        raise ConfigurationError(
            f"Unknown model '{model}' for provider '{provider}'. Available: {available}"
        )

    _PROVIDER_CLASSES: dict[str, type[LLM]] = {
        "openai": OpenAI,
        "anthropic": Anthropic,
        "gemini": Gemini,
        "deepseek": DeepSeek,
        "cohere": Cohere,
    }
    cls = _PROVIDER_CLASSES.get(provider)
    if cls is None:
        raise ConfigurationError(f"Unknown LLM provider '{provider}'")

    provider_kwargs: dict[str, Any] = {}
    if provider == "deepseek":
        provider_kwargs = {
            "reasoning_effort": model_attributes.get("reasoning_effort"),
            "thinking": model_attributes.get("thinking"),
        }

    llm = cls(
        model=model,
        input_cost=model_attributes["input_cost"],
        output_cost=model_attributes["output_cost"],
        supports_temperature_top_p=model_attributes.get("supports_temperature_top_p", True),
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
        **provider_kwargs,
    )

    if deprecation_warning:
        llm.deprecation_warning = deprecation_warning
        llm.requested_model = requested_model

    return llm


def get_all_llm_instances() -> Iterator[LLM]:
    """Create LLM instances for all configured providers and models.

    Yields LLM instances one at a time, which is useful for initialization
    or testing all available models.

    Yields:
        LLM instances for each configured provider/model combination.

    Example:
        >>> for llm in get_all_llm_instances():
        ...     print(llm.get_full_model_name())
    """
    for provider, provider_config in LLM_CONFIG.items():
        models = provider_config.get("models", {})
        for model in models:
            logger.debug("Creating LLM instance: %s/%s", provider, model)
            yield get_llm_instance(provider, model)


# ---------------------------------------------------------------------------
# Alias functions
# ---------------------------------------------------------------------------


def get_llm_by_alias(
    alias: str,
    *,
    base_url: str | None = None,
    default_headers: dict[str, str] | None = None,
) -> LLM:
    """Create an LLM instance (or cascade) from a registered alias.

    Args:
        alias: The alias name (e.g., "fast", "thinking").
        base_url: Optional custom base URL for routing through a proxy.
        default_headers: Optional headers sent with every request.

    Returns:
        An LLM instance for single-provider aliases, or an LLMCascade for
        cascade aliases.

    Raises:
        ConfigurationError: If the alias is not found.

    Example:
        >>> llm = get_llm_by_alias("fast")
        >>> response = await llm.get_response("Hello!")
    """
    target = _ALIAS_REGISTRY.get(alias)
    if target is None:
        available = ", ".join(sorted(_ALIAS_REGISTRY.keys())) or "(none)"
        raise ConfigurationError(f"Unknown alias '{alias}'. Available: {available}")

    if isinstance(target, list):
        # Lazy import to avoid circular dependency (cascade.py imports from factory.py)
        from majordomo_llm.cascade import LLMCascade

        return LLMCascade(
            target,
            base_url=base_url,
            default_headers=default_headers,
        )
    else:
        provider, model = target
        return get_llm_instance(provider, model, base_url=base_url, default_headers=default_headers)


def register_alias(name: str, target: AliasTarget) -> None:
    """Register or override a model alias.

    Args:
        name: The alias name (e.g., "fast", "thinking").
        target: Either a (provider, model) tuple for a single model,
            or a list of (provider, model) tuples for a cascade.

    Raises:
        ConfigurationError: If a referenced provider or model is not in LLM_CONFIG,
            or if the target format is invalid.

    Example:
        >>> register_alias("fast", ("anthropic", "claude-3-5-haiku-20241022"))
        >>> register_alias("resilient", [
        ...     ("anthropic", "claude-sonnet-4-20250514"),
        ...     ("openai", "gpt-4o"),
        ... ])
    """
    if isinstance(target, list):
        if len(target) < 2:
            raise ConfigurationError(f"Cascade alias '{name}' must have at least 2 providers")
        for provider, model in target:
            _validate_provider_model(provider, model, name)
    elif isinstance(target, tuple) and len(target) == 2:
        _validate_provider_model(target[0], target[1], name)
    else:
        raise ConfigurationError(
            f"Alias target must be a (provider, model) tuple or list of tuples, "
            f"got {type(target).__name__}"
        )
    _ALIAS_REGISTRY[name] = target


def unregister_alias(name: str) -> None:
    """Remove a previously registered alias.

    Args:
        name: The alias name to remove.

    Raises:
        ConfigurationError: If the alias does not exist.
    """
    if name not in _ALIAS_REGISTRY:
        available = ", ".join(sorted(_ALIAS_REGISTRY.keys())) or "(none)"
        raise ConfigurationError(f"Unknown alias '{name}'. Available: {available}")
    del _ALIAS_REGISTRY[name]


def clear_aliases() -> None:
    """Remove all registered aliases (both YAML-loaded and programmatic)."""
    _ALIAS_REGISTRY.clear()


def get_aliases() -> dict[str, AliasTarget]:
    """Return a copy of all currently registered aliases."""
    return dict(_ALIAS_REGISTRY)
