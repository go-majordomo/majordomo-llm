"""Tests for the BedrockMantle provider.

BedrockMantle subclasses Anthropic and only overrides ``__init__`` — auth
resolution (AWS_BEARER_TOKEN_BEDROCK), endpoint construction from region, and
provider identity. The actual request methods are inherited and exercised by
the existing Anthropic tests, so these tests focus on the init-time
divergences.
"""

from unittest.mock import patch

import pytest

from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers import BedrockMantle


class TestBedrockMantleInit:
    def test_raises_when_no_bearer_token(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                BedrockMantle(
                    model="anthropic.claude-opus-4-7",
                    input_cost=5.0,
                    output_cost=25.0,
                    region="us-east-1",
                )
            assert "AWS_BEARER_TOKEN_BEDROCK" in str(exc_info.value)

    def test_raises_when_no_region_and_no_base_url(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                BedrockMantle(
                    model="anthropic.claude-opus-4-7",
                    input_cost=5.0,
                    output_cost=25.0,
                    api_key="test-key",
                )
            assert "region" in str(exc_info.value).lower()

    def test_region_from_env_var(self):
        with patch.dict(
            "os.environ",
            {"AWS_REGION": "us-west-2", "AWS_BEARER_TOKEN_BEDROCK": "test-key"},
            clear=True,
        ):
            llm = BedrockMantle(
                model="anthropic.claude-opus-4-7",
                input_cost=5.0,
                output_cost=25.0,
            )
            assert llm.region == "us-west-2"
            # The Anthropic client is built with the regionalized Mantle endpoint.
            assert llm.client.base_url.host == "bedrock-mantle.us-west-2.api.aws"

    def test_falls_back_to_aws_default_region(self):
        with patch.dict(
            "os.environ",
            {"AWS_DEFAULT_REGION": "eu-west-1", "AWS_BEARER_TOKEN_BEDROCK": "test-key"},
            clear=True,
        ):
            llm = BedrockMantle(
                model="anthropic.claude-opus-4-7",
                input_cost=5.0,
                output_cost=25.0,
            )
            assert llm.region == "eu-west-1"
            assert llm.client.base_url.host == "bedrock-mantle.eu-west-1.api.aws"

    def test_provider_identity_is_bedrock_mantle(self):
        """Cost tracking, cascade dispatch, and logs must see this as its own
        provider — not as vanilla Anthropic."""
        llm = BedrockMantle(
            model="anthropic.claude-opus-4-7",
            input_cost=5.0,
            output_cost=25.0,
            api_key="test-key",
            region="us-east-1",
        )
        assert llm.provider == "bedrock_mantle"

    def test_endpoint_built_from_region(self):
        llm = BedrockMantle(
            model="anthropic.claude-opus-4-7",
            input_cost=5.0,
            output_cost=25.0,
            api_key="test-key",
            region="ap-southeast-2",
        )
        assert llm.client.base_url.host == "bedrock-mantle.ap-southeast-2.api.aws"
        assert "/anthropic" in str(llm.client.base_url)

    def test_explicit_base_url_overrides_region(self):
        """An explicit base_url wins — region-derived endpoint is only the
        default. Lets users point at a proxy (e.g. Steward) without changing
        their AWS region wiring."""
        llm = BedrockMantle(
            model="anthropic.claude-opus-4-7",
            input_cost=5.0,
            output_cost=25.0,
            api_key="test-key",
            region="us-east-1",
            base_url="https://gateway.example.com",
        )
        assert llm.client.base_url.host == "gateway.example.com"

    def test_stores_model_and_costs(self):
        llm = BedrockMantle(
            model="anthropic.claude-opus-4-7",
            input_cost=5.0,
            output_cost=25.0,
            api_key="test-key",
            region="us-east-1",
        )
        assert llm.model == "anthropic.claude-opus-4-7"
        assert llm.input_cost == 5.0
        assert llm.output_cost == 25.0

    def test_injects_steward_routing_headers_when_proxying(self):
        """When a base_url is set (proxy routing), the provider auto-injects
        the metadata Steward needs to route Mantle traffic:
        - ``x-majordomo-provider: bedrock-mantle`` disambiguates from vanilla
          Anthropic traffic (both speak the Messages API shape).
        - ``X-Majordomo-Bedrock-Region`` tells Steward which AWS region to
          forward to (Mantle is region-pinned upstream).
        """
        llm = BedrockMantle(
            model="anthropic.claude-opus-4-7",
            input_cost=5.0,
            output_cost=25.0,
            api_key="test-key",
            region="us-west-2",
            base_url="https://gateway.example.com",
        )
        assert llm.default_headers["x-majordomo-provider"] == "bedrock-mantle"
        assert llm.default_headers["X-Majordomo-Bedrock-Region"] == "us-west-2"

    def test_does_not_inject_steward_headers_when_direct(self):
        """Direct AWS calls hit the regional Mantle endpoint already — no
        proxy needs the routing or region hints. Regression guard against
        leaking Majordomo metadata to AWS."""
        llm = BedrockMantle(
            model="anthropic.claude-opus-4-7",
            input_cost=5.0,
            output_cost=25.0,
            api_key="test-key",
            region="us-east-1",
        )
        headers = llm.default_headers or {}
        assert "X-Majordomo-Bedrock-Region" not in headers
        assert "x-majordomo-provider" not in headers

    def test_user_default_headers_preserved_alongside_region(self):
        """User-provided headers must coexist with the auto-injected region —
        and explicit user values win on key collision."""
        llm = BedrockMantle(
            model="anthropic.claude-opus-4-7",
            input_cost=5.0,
            output_cost=25.0,
            api_key="test-key",
            region="us-east-1",
            base_url="https://gateway.example.com",
            default_headers={
                "X-Majordomo-Key": "mk-1",
                "X-Majordomo-Bedrock-Region": "eu-west-1",  # explicit override
            },
        )
        # Caller-provided values take precedence.
        assert llm.default_headers["X-Majordomo-Bedrock-Region"] == "eu-west-1"
        assert llm.default_headers["X-Majordomo-Key"] == "mk-1"

    def test_inherits_anthropic_client_type(self):
        """The client must be an anthropic.AsyncAnthropic — that's how we
        inherit Claude's full feature set (structured outputs, prompt caching,
        extended thinking) without reimplementing anything."""
        import anthropic

        llm = BedrockMantle(
            model="anthropic.claude-opus-4-7",
            input_cost=5.0,
            output_cost=25.0,
            api_key="test-key",
            region="us-east-1",
        )
        assert isinstance(llm.client, anthropic.AsyncAnthropic)
