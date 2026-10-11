"""Tests for the factory module."""

from unittest.mock import patch

import pytest

from majordomo_llm import (
    LLM_CONFIG,
    get_aliases,
    get_all_llm_instances,
    get_llm_instance,
    get_supported_providers,
)
from majordomo_llm.base import DEFAULT_STREAM_MAX_TOKENS
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers.anthropic import Anthropic
from majordomo_llm.providers.gemini import Gemini
from majordomo_llm.providers.openai import OpenAI


@pytest.fixture
def mock_all_clients():
    """Mock all provider API clients and environment variables."""
    env_vars = {
        "ANTHROPIC_API_KEY": "test-key",
        "OPENAI_API_KEY": "test-key",
        "GEMINI_API_KEY": "test-key",
        "CO_API_KEY": "test-key",
        "AWS_BEARER_TOKEN_BEDROCK": "test-key",
        "AWS_REGION": "us-east-1",
        "FIREWORKS_API_KEY": "test-key",
        "TOGETHER_API_KEY": "test-key",
        "BASETEN_API_KEY": "test-key",
    }
    with (
        patch.dict("os.environ", env_vars),
        patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"),
        patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"),
        patch("majordomo_llm.providers.gemini.genai.Client"),
        patch("majordomo_llm.providers.cohere.cohere.AsyncClientV2"),
        patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"),
        patch("majordomo_llm.providers.together.openai.AsyncOpenAI"),
        # Baseten builds its client in the shared OpenAI-compatible base.
        patch("majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"),
    ):
        yield


class TestGetLLMInstance:
    """Tests for get_llm_instance factory function."""

    def test_creates_anthropic_provider(self, mock_all_clients):
        """Should create Anthropic instance for anthropic provider."""
        llm = get_llm_instance("anthropic", "claude-sonnet-5")

        assert isinstance(llm, Anthropic)
        assert llm.provider == "anthropic"
        assert llm.model == "claude-sonnet-5"

    def test_creates_openai_provider(self, mock_all_clients):
        """Should create OpenAI instance for openai provider."""
        llm = get_llm_instance("openai", "gpt-6.1-sol")

        assert isinstance(llm, OpenAI)
        assert llm.provider == "openai"
        assert llm.model == "gpt-6.1-sol"

    def test_creates_gemini_provider(self, mock_all_clients):
        """Should create Gemini instance for gemini provider."""
        llm = get_llm_instance("gemini", "gemini-3.8-flash")

        assert isinstance(llm, Gemini)
        assert llm.provider == "gemini"
        assert llm.model == "gemini-3.8-flash"

    def test_sets_correct_costs_from_config(self, mock_all_clients):
        """Should set input/output costs from LLM_CONFIG."""
        llm = get_llm_instance("anthropic", "claude-sonnet-5")

        expected_config = LLM_CONFIG["anthropic"]["models"]["claude-sonnet-5"]
        assert llm.input_cost == expected_config["input_cost"]
        assert llm.output_cost == expected_config["output_cost"]

    def test_yaml_model_override_resolves_to_upstream_id(self, mock_all_clients):
        """Profile entries with ``model:`` override should pass the upstream ID
        to the provider while keeping the YAML key as the lookup name."""
        llm = get_llm_instance("anthropic", "claude-opus-4-8-medium")
        assert llm.model == "claude-opus-4-8"
        assert llm.reasoning_effort == "medium"
        assert llm.thinking == "adaptive"

        llm = get_llm_instance("anthropic", "claude-opus-4-8-deep")
        assert llm.model == "claude-opus-4-8"
        assert llm.reasoning_effort == "xhigh"
        assert llm.thinking == "adaptive"

    def test_sets_supports_temperature_top_p_flag(self, mock_all_clients):
        """Should set supports_temperature_top_p from config."""
        # Model with flag set to False
        llm = get_llm_instance("anthropic", "claude-sonnet-5-5")
        assert llm.supports_temperature_top_p is False

        # Model without flag (defaults to True)
        llm = get_llm_instance("gemini", "gemini-3.8-flash")
        assert llm.supports_temperature_top_p is True

    def test_raises_for_unknown_provider(self):
        """Should raise ConfigurationError for unknown provider."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_llm_instance("unknown_provider", "some-model")

        assert "Unknown LLM provider" in str(exc_info.value)
        assert "unknown_provider" in str(exc_info.value)

    def test_raises_for_unknown_model(self):
        """Should raise ConfigurationError for unknown model."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_llm_instance("anthropic", "unknown-model")

        assert "Unknown model" in str(exc_info.value)
        assert "unknown-model" in str(exc_info.value)

    def test_replaces_deprecated_model_with_warning(self, mock_all_clients):
        """Should auto-replace a deprecated model and log a warning."""
        llm = get_llm_instance("gemini", "gemini-2.5-flash")

        assert isinstance(llm, Gemini)
        assert llm.model == "gemini-3.8-flash"

    def test_replaces_deprecated_anthropic_model(self, mock_all_clients):
        """Should auto-replace deprecated Anthropic models."""
        llm = get_llm_instance("anthropic", "claude-sonnet-4-5-20250929")

        assert isinstance(llm, Anthropic)
        assert llm.model == "claude-sonnet-5-5"

    def test_deprecated_model_sets_warning_and_requested_model(self, mock_all_clients):
        """Should set deprecation_warning and requested_model on the LLM instance."""
        llm = get_llm_instance("gemini", "gemini-2.5-flash")

        assert llm.requested_model == "gemini-2.5-flash"
        assert llm.deprecation_warning is not None
        assert "gemini-2.5-flash" in llm.deprecation_warning
        assert "gemini-3.8-flash" in llm.deprecation_warning

    def test_non_deprecated_model_has_no_warning(self, mock_all_clients):
        """Should not set deprecation info for active models."""
        llm = get_llm_instance("openai", "gpt-6.1-sol")

        assert llm.deprecation_warning is None
        assert llm.requested_model is None


class TestGetAllLLMInstances:
    """Tests for get_all_llm_instances function."""

    def test_yields_instances_for_all_configured_models(self, mock_all_clients):
        """Should yield an LLM instance for each directly-callable model."""
        instances = list(get_all_llm_instances())

        # Count expected models, excluding gateway-only providers (majordomo),
        # which are not directly instantiable and are skipped by enumeration.
        expected_count = sum(
            len(provider_config["models"])
            for provider, provider_config in LLM_CONFIG.items()
            if provider != "majordomo"
        )

        assert len(instances) == expected_count

    def test_yields_correct_provider_types(self, mock_all_clients):
        """Should yield correct provider types."""
        instances = list(get_all_llm_instances())

        providers = {llm.provider for llm in instances}
        assert providers == {
            "openai",
            "anthropic",
            "gemini",
            "cohere",
            "bedrock",
            "bedrock_mantle",
            "fireworks",
            "together",
            "baseten",
        }


class TestLLMConfig:
    """Tests for LLM_CONFIG structure."""

    def test_all_providers_have_models(self):
        """Each provider should have at least one model configured."""
        for provider, config in LLM_CONFIG.items():
            assert "models" in config, f"{provider} missing 'models' key"
            assert len(config["models"]) > 0, f"{provider} has no models"

    def test_all_models_have_required_costs(self):
        """Each model should have input_cost and output_cost.

        Gateway-only providers (majordomo) are exempt: their cost is resolved
        per request from the backend the gateway routes to, so their config
        entries intentionally carry no token costs.
        """
        for provider, config in LLM_CONFIG.items():
            if provider == "majordomo":
                continue
            for model, model_config in config["models"].items():
                assert "input_cost" in model_config, f"{provider}/{model} missing input_cost"
                assert "output_cost" in model_config, f"{provider}/{model} missing output_cost"
                assert model_config["input_cost"] >= 0, f"{provider}/{model} invalid input_cost"
                assert model_config["output_cost"] >= 0, f"{provider}/{model} invalid output_cost"


class TestUseWebSearchForwarding:
    """Tests for use_web_search forwarding from the factory."""

    def test_forwards_to_supported_provider(self, mock_all_clients):
        llm = get_llm_instance("anthropic", "claude-sonnet-5-5", use_web_search=True)
        assert llm.use_web_search is True

    def test_default_is_false(self, mock_all_clients):
        llm = get_llm_instance("anthropic", "claude-sonnet-5-5")
        assert llm.use_web_search is False

    def test_raises_on_unsupported_model(self, mock_all_clients):
        with pytest.raises(ConfigurationError) as exc_info:
            get_llm_instance("openai", "gpt-5-nano", use_web_search=True)
        assert "does not support web search" in str(exc_info.value)

    def test_silently_ignored_for_unsupported_provider(self, mock_all_clients):
        # Cohere does not implement web search. The factory should accept
        # use_web_search=True without raising and not forward the flag — the
        # resulting instance should still report use_web_search=False.
        llm = get_llm_instance("cohere", "command-a-03-2025", use_web_search=True)
        assert llm.use_web_search is False


class TestUsePromptCachingForwarding:
    """Tests for use_prompt_caching forwarding/override from the factory."""

    def test_default_is_true_for_anthropic(self, mock_all_clients):
        llm = get_llm_instance("anthropic", "claude-sonnet-5-5")
        assert llm.use_prompt_caching is True

    def test_override_disables_caching(self, mock_all_clients):
        llm = get_llm_instance(
            "anthropic", "claude-sonnet-5-5", use_prompt_caching=False
        )
        assert llm.use_prompt_caching is False

    def test_override_forwarded_to_bedrock_mantle(self, mock_all_clients):
        llm = get_llm_instance(
            "bedrock_mantle",
            "anthropic.claude-haiku-4-5",
            use_prompt_caching=False,
            region="us-east-1",
        )
        assert llm.use_prompt_caching is False

    def test_cache_costs_loaded_from_config(self, mock_all_clients):
        llm = get_llm_instance("anthropic", "claude-sonnet-5-5")
        assert llm.cached_input_cost == 0.20
        assert llm.cache_write_cost == 2.50


class TestStructuredOutputForwarding:
    """Tests for supports_structured_outputs on OpenAI-compatible providers."""

    def test_defaults_to_true(self, mock_all_clients):
        llm = get_llm_instance("baseten", "moonshotai/Kimi-K3")
        assert llm.supports_structured_outputs is True

    def test_config_can_opt_out(self, mock_all_clients):
        # Baseten's DeepSeek V4 Pro accepted json_schema and then ignored it.
        llm = get_llm_instance("baseten", "deepseek-ai/DeepSeek-V4-Pro-0813")
        assert llm.supports_structured_outputs is False


class TestFireworksReasoningDefaults:
    """Fireworks reasons at its top tier unless told otherwise."""

    @pytest.mark.parametrize(
        "model",
        [
            # Thinking-only on Fireworks: it cannot be disabled, only lowered.
            "accounts/fireworks/models/glm-5p3",
            "accounts/fireworks/models/kimi-k3",
            "accounts/fireworks/models/qwen3p8-2p4t-a95b",
            "accounts/fireworks/models/qwen3p8-max",
        ],
    )
    def test_controllable_models_are_pinned_to_low_effort(self, model, mock_all_clients):
        llm = get_llm_instance("fireworks", model)
        assert llm.reasoning_effort == "low"
        assert llm.thinking is None
        assert llm._fireworks_request_kwargs() == {"reasoning_effort": "low"}

    def test_inkling_sends_no_reasoning_option(self, mock_all_clients):
        # No setting changes Inkling's reasoning, so none is sent.
        llm = get_llm_instance("fireworks", "accounts/fireworks/models/inkling")
        assert llm._fireworks_request_kwargs() == {}


class TestOpenWeightReasoningDefaults:
    """Together and Baseten: reasoning_effort is set only where low lowers it."""

    LOW = (
        ("together", "zai-org/GLM-5.3"),
        ("together", "zai-org/GLM-5.3-Flash"),
        ("together", "moonshotai/Kimi-K3"),
        ("together", "Qwen/Qwen3.8-2.4T-A95B"),
        ("baseten", "zai-org/GLM-5.3"),
        ("baseten", "zai-org/GLM-5.3-Flash"),
        ("baseten", "moonshotai/Kimi-K3"),
    )

    #: Effort has no effect on these, or (Baseten GLM-5.2) switches reasoning ON.
    UNSET = (
        ("together", "deepseek-ai/DeepSeek-V4-Pro-0813"),
        ("together", "deepseek-ai/DeepSeek-V4.1-Flash"),
        ("together", "zai-org/GLM-5.2"),
        ("together", "Qwen/Qwen3.5-9B"),
        ("together", "thinkingmachines/Inkling"),
        ("baseten", "deepseek-ai/DeepSeek-V4-Pro-0813"),
        ("baseten", "deepseek-ai/DeepSeek-V4.1-Flash"),
        ("baseten", "zai-org/GLM-5.2"),
    )

    @pytest.mark.parametrize(("provider", "model"), LOW)
    def test_pinned_to_low_effort(self, provider, model, mock_all_clients):
        llm = get_llm_instance(provider, model)
        assert llm.reasoning_effort == "low"
        assert llm.thinking is None

    @pytest.mark.parametrize(("provider", "model"), UNSET)
    def test_sends_no_reasoning_option(self, provider, model, mock_all_clients):
        llm = get_llm_instance(provider, model)
        assert llm.reasoning_effort is None
        assert llm.thinking is None


class TestReasoningOverrides:
    """get_llm_instance(reasoning_effort=, thinking=) replaces the config values."""

    FIREWORKS_GLM = ("fireworks", "accounts/fireworks/models/glm-5p3")  # config: low

    def test_config_values_apply_when_nothing_is_passed(self, mock_all_clients):
        llm = get_llm_instance(*self.FIREWORKS_GLM)
        assert (llm.reasoning_effort, llm.thinking) == ("low", None)

    def test_reasoning_effort_overrides_config(self, mock_all_clients):
        llm = get_llm_instance(*self.FIREWORKS_GLM, reasoning_effort="high")
        assert (llm.reasoning_effort, llm.thinking) == ("high", None)

    def test_thinking_alone_drops_the_configured_effort(self, mock_all_clients):
        """Passing either argument replaces both, so a config effort is never
        sent alongside a caller's thinking mode."""
        llm = get_llm_instance("together", "moonshotai/Kimi-K3", thinking="disabled")
        assert (llm.reasoning_effort, llm.thinking) == (None, "disabled")
        assert llm._together_request_kwargs() == {"extra_body": {"thinking": {"type": "disabled"}}}

    def test_default_sends_no_reasoning_option(self, mock_all_clients):
        llm = get_llm_instance(*self.FIREWORKS_GLM, reasoning_effort="default")
        assert (llm.reasoning_effort, llm.thinking) == (None, None)
        assert llm._fireworks_request_kwargs() == {}

    def test_override_reaches_a_model_with_no_configured_value(self, mock_all_clients):
        # Inkling on Together only stops reasoning with effort "none".
        llm = get_llm_instance("together", "thinkingmachines/Inkling", reasoning_effort="none")
        assert llm._together_request_kwargs() == {"reasoning_effort": "none"}

    def test_baseten_accepts_the_override(self, mock_all_clients):
        llm = get_llm_instance(
            "baseten", "deepseek-ai/DeepSeek-V4.1-Flash", thinking="disabled"
        )
        assert (llm.reasoning_effort, llm.thinking) == (None, "disabled")

    def test_anthropic_profile_is_replaced_as_a_whole(self, mock_all_clients):
        # claude-opus-4-8-medium is configured thinking: adaptive + effort: medium.
        llm = get_llm_instance("anthropic", "claude-opus-4-8-medium", reasoning_effort="max")
        assert (llm.reasoning_effort, llm.thinking) == ("max", None)

    def test_invalid_value_is_rejected_by_the_provider(self, mock_all_clients):
        with pytest.raises(ValueError, match="reasoning_effort"):
            get_llm_instance(*self.FIREWORKS_GLM, reasoning_effort="turbo")

    @pytest.mark.parametrize(
        ("provider", "model"),
        [
            ("openai", "gpt-6.1-sol"),
            ("gemini", "gemini-3.8-flash"),
            ("cohere", "command-a-03-2025"),
            ("bedrock_mantle", "anthropic.claude-opus-5"),
        ],
    )
    def test_unsupported_provider_raises(self, provider, model, mock_all_clients):
        with pytest.raises(ConfigurationError, match="does not accept reasoning_effort"):
            get_llm_instance(provider, model, reasoning_effort="low")


class TestForcedToolChoiceForwarding:
    """Tests for supports_forced_tool_choice on the Anthropic-shaped providers."""

    NO_FORCED_TOOL_CHOICE = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1")

    def test_new_generation_opts_out(self, mock_all_clients):
        for model in self.NO_FORCED_TOOL_CHOICE:
            llm = get_llm_instance("anthropic", model)
            assert llm.model == model
            assert llm.supports_forced_tool_choice is False, model
            assert llm.supports_structured_outputs is True, model

    def test_defaults_to_true(self, mock_all_clients):
        for model in LLM_CONFIG["anthropic"]["models"]:
            if model in self.NO_FORCED_TOOL_CHOICE:
                continue
            assert get_llm_instance("anthropic", model).supports_forced_tool_choice is True, model

    def test_bedrock_mantle_defaults_to_true(self, mock_all_clients):
        for model in LLM_CONFIG["bedrock_mantle"]["models"]:
            llm = get_llm_instance("bedrock_mantle", model, region="us-east-1")
            assert llm.supports_forced_tool_choice is True, model


class TestLatestModelsRegistered:
    """The current generation from each first-party vendor resolves from config."""

    @pytest.mark.parametrize(
        ("provider", "model", "input_cost", "output_cost", "cached_input_cost"),
        [
            ("anthropic", "claude-opus-5-5", 4.00, 20.00, 0.20),
            ("anthropic", "claude-sonnet-5-5", 2.00, 10.00, 0.20),
            ("anthropic", "claude-fable-5-1", 10.00, 50.00, 0.25),
            ("anthropic", "claude-haiku-5-5", 0.10, 0.50, 0.01),
            ("openai", "gpt-6-astra", 10.00, 50.00, 1.00),
            ("openai", "gpt-6.1-sol", 2.00, 10.00, 0.10),
            ("openai", "gpt-6-sol", 2.00, 10.00, 0.20),
            ("openai", "gpt-6-luna", 0.10, 0.50, 0.01),
            ("gemini", "gemini-3.8-flash", 0.75, 3.75, 0.075),
            ("gemini", "gemini-3.7-flash", 0.75, 3.75, 0.075),
            ("fireworks", "accounts/fireworks/models/glm-5p3", 1.40, 4.40, 0.26),
            ("fireworks", "accounts/fireworks/models/qwen3p8-2p4t-a95b", 2.00, 6.00, 0.25),
            ("fireworks", "accounts/fireworks/models/qwen3p8-max", 2.00, 6.00, 0.25),
            ("together", "deepseek-ai/DeepSeek-V4-Pro-0813", 1.32, 3.96, 0.13),
            ("together", "deepseek-ai/DeepSeek-V4.1-Flash", 0.30, 1.20, 0.006),
            ("together", "zai-org/GLM-5.3", 1.40, 4.40, 0.26),
            ("together", "zai-org/GLM-5.3-Flash", 0.15, 0.50, 0.03),
            ("baseten", "zai-org/GLM-5.3", 1.40, 4.40, 0.14),
            ("baseten", "zai-org/GLM-5.3-Flash", 0.15, 0.50, 0.03),
            ("baseten", "deepseek-ai/DeepSeek-V4-Pro-0813", 1.32, 3.96, 0.132),
            ("baseten", "deepseek-ai/DeepSeek-V4.1-Flash", 0.30, 1.20, 0.007),
            ("together", "Qwen/Qwen3.8-2.4T-A95B", 2.00, 6.00, 0.50),
            ("together", "Qwen/Qwen3.7-Max", 2.50, 7.50, 0.50),
            ("together", "Qwen/Qwen3.7-Plus", 0.32, 1.28, None),
            ("together", "Qwen/Qwen3.8-Flash", 0.15, 0.47, None),
        ],
    )
    def test_resolves_with_published_rates(
        self, mock_all_clients, provider, model, input_cost, output_cost, cached_input_cost
    ):
        llm = get_llm_instance(provider, model)
        assert llm.model == model
        assert llm.input_cost == input_cost
        assert llm.output_cost == output_cost
        assert llm.cached_input_cost == cached_input_cost


class TestGatewayProviderIsOptIn:
    """The majordomo provider must never be reached by default.

    It is a routing pseudo-provider: it names a canonical model and lets the
    gateway pick a backend, so it cannot run without a live Steward instance.
    Anything that enumerates providers has to skip it, and it must fail loudly
    rather than silently when reached without a gateway URL.

    This is distinct from routing a CONCRETE provider through the gateway for
    usage tracking, which is just a base_url plus headers and stays available.
    """

    def test_excluded_from_sweeps(self, mock_all_clients):
        providers = {llm.provider for llm in get_all_llm_instances()}
        assert "majordomo" not in providers

    def test_still_listed_as_supported(self):
        # Excluded from sweeps, but callable on purpose — it must stay
        # discoverable so an opt-in caller can find it.
        assert "majordomo" in get_supported_providers()

    def test_requires_explicit_gateway_url(self, mock_all_clients):
        with pytest.raises(ConfigurationError, match="requires base_url"):
            get_llm_instance("majordomo", "glm-5.2")

    def test_no_alias_resolves_to_it(self):
        for name, target in get_aliases().items():
            hops = target if isinstance(target, list) else [target]
            for provider, _ in hops:
                assert provider != "majordomo", (
                    f"alias {name!r} would route to the gateway provider by default"
                )


class TestMaxTokensForwarding:
    """max_tokens is pinned only where a model cannot take the library default.

    The defaults are 16000 non-streaming / 64000 streaming. A model whose real
    ceiling is at or above those inherits them and must NOT pin a value: pinning
    the vendor ceiling instead makes it the per-request default, and on Anthropic
    anything over 21333 is rejected by the SDK before the request is sent.
    """

    def test_anthropic_models_pin_nothing(self, mock_all_clients):
        """Every Anthropic ceiling is >= the streaming default, so none pins."""
        for model in LLM_CONFIG["anthropic"]["models"]:
            assert get_llm_instance("anthropic", model).max_tokens is None, model

    def test_bedrock_mantle_pins_nothing(self, mock_all_clients):
        for model in LLM_CONFIG["bedrock_mantle"]["models"]:
            llm = get_llm_instance("bedrock_mantle", model, region="us-east-1")
            assert llm.max_tokens is None, model

    def test_bedrock_pins_only_the_low_ceiling_models(self, mock_all_clients):
        """Llama 4 (8192) and DeepSeek-R1 (32768) sit below the 64000 stream default."""
        pinned = {
            m: a["max_tokens"]
            for m, a in LLM_CONFIG["bedrock"]["models"].items()
            if "max_tokens" in a
        }
        assert pinned == {
            "us.meta.llama4-maverick-17b-instruct-v1:0": 8192,
            "us.meta.llama4-scout-17b-instruct-v1:0": 8192,
            "us.deepseek.r1-v1:0": 32768,
        }

    def test_pinned_value_reaches_the_instance(self, mock_all_clients):
        llm = get_llm_instance("bedrock", "us.deepseek.r1-v1:0", region="us-east-1")
        assert llm.max_tokens == 32768

    def test_no_pin_exceeds_the_streaming_default(self):
        """A pin above the default is pointless; above 21333 it breaks Anthropic."""
        for provider in ("anthropic", "bedrock", "bedrock_mantle"):
            for model, attrs in LLM_CONFIG[provider]["models"].items():
                pinned = attrs.get("max_tokens")
                if pinned is not None:
                    assert pinned < DEFAULT_STREAM_MAX_TOKENS, f"{provider}/{model}"

    def test_not_forwarded_to_providers_that_ignore_it(self, mock_all_clients):
        """Providers that send no cap inherit the model default, not ours."""
        assert get_llm_instance("openai", "gpt-6.1-sol").max_tokens is None
        assert get_llm_instance("gemini", "gemini-3.8-flash").max_tokens is None


class TestDeprecatedModels:
    """deprecated_models holds only vendor-withdrawn models with a named successor."""

    MAPPED = {
        ("anthropic", "claude-sonnet-4-5-20250929"): "claude-sonnet-5-5",
        ("gemini", "gemini-2.5-pro"): "gemini-3.1-pro-preview",
        ("gemini", "gemini-2.5-flash"): "gemini-3.8-flash",
        ("gemini", "gemini-2.5-flash-lite"): "gemini-3.5-flash-lite",
    }

    #: Removed outright on 2026-10-11, with no mapping. A sample of each group.
    REMOVED = (
        ("openai", "gpt-5.4"),
        ("openai", "gpt-4.1"),
        ("openai", "gpt-4o"),
        ("openai", "o3"),
        ("anthropic", "claude-opus-4-7"),
        ("anthropic", "claude-sonnet-4-6"),
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("anthropic", "claude-opus-4-1-20250805"),
        ("anthropic", "claude-3-5-haiku-20241022"),
        ("gemini", "gemini-2.0-flash"),
    )

    def test_the_mapping_is_exactly_the_documented_set(self):
        from majordomo_llm.factory import _DEPRECATED_MODELS

        actual = {
            (provider, old): new
            for provider, mapping in _DEPRECATED_MODELS.items()
            for old, new in mapping.items()
        }
        assert actual == self.MAPPED

    def test_mapped_models_are_absent_from_the_models_block(self):
        """The factory only consults deprecated_models when a model is absent."""
        still_registered = [
            key for key in self.MAPPED if key[1] in LLM_CONFIG[key[0]]["models"]
        ]
        assert still_registered == []

    @pytest.mark.parametrize(("key", "replacement"), sorted(MAPPED.items()))
    def test_resolves_to_replacement(self, key, replacement, mock_all_clients):
        provider, model = key
        llm = get_llm_instance(provider, model)
        assert llm.model == replacement
        assert llm.requested_model == model
        assert model in llm.deprecation_warning

    def test_every_mapping_targets_a_registered_model(self):
        """Mappings are not chained, so a replacement that is itself removed
        leaves the old name unresolvable."""
        dangling = [
            key for key, new in self.MAPPED.items() if new not in LLM_CONFIG[key[0]]["models"]
        ]
        assert dangling == []

    @pytest.mark.parametrize(("provider", "model"), REMOVED)
    def test_removed_models_raise_unknown_model(self, provider, model, mock_all_clients):
        with pytest.raises(ConfigurationError, match="Unknown model"):
            get_llm_instance(provider, model)

    def test_no_alias_targets_a_deprecated_or_removed_model(self):
        """Alias validation resolves against the models block only, not
        deprecated_models — a YAML alias naming a retired model raises
        ConfigurationError at import and takes the whole package down with it."""
        gone = {model for _provider, model in (*self.MAPPED, *self.REMOVED)}
        for name, target in get_aliases().items():
            members = target if isinstance(target, list) else [target]
            for _provider, model in members:
                assert model not in gone, f"alias {name!r} still targets {model!r}"
