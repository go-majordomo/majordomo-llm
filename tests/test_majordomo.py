"""Tests for the Majordomo gateway provider (optimal routing)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from majordomo_llm.base import TOKENS_PER_MILLION
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers import Majordomo

GATEWAY_URL = "http://localhost:7680"

# Routed backends the gateway may select for the canonical "glm-5.2" model.
# Their cached-read rates differ (0.14 vs 0.26), which is the whole point of
# resolving cost from the routed pair rather than a fixed config entry.
FIREWORKS_GLM = ("fireworks", "accounts/fireworks/models/glm-5p2")
TOGETHER_GLM = ("together", "zai-org/GLM-5.2")


class CountryInfo(BaseModel):
    """Test model for structured responses."""

    name: str
    capital: str
    population: int


COUNTRY_SCHEMA = CountryInfo.model_json_schema()


def _make_raw(
    *,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    routed_provider: str | None = None,
    routed_model: str | None = None,
):
    """Build a mock ``with_raw_response`` result (headers + parsed body)."""
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = content
    completion.usage.prompt_tokens = prompt_tokens
    completion.usage.completion_tokens = completion_tokens
    if cached_tokens:
        completion.usage.prompt_tokens_details.cached_tokens = cached_tokens
    else:
        completion.usage.prompt_tokens_details = None

    headers: dict[str, str] = {}
    if routed_provider is not None:
        headers["X-Majordomo-Routed-Provider"] = routed_provider
    if routed_model is not None:
        headers["X-Majordomo-Routed-Model"] = routed_model

    raw = MagicMock()
    raw.headers = headers
    raw.parse = MagicMock(return_value=completion)
    return raw


@pytest.fixture
def majordomo_llm():
    """Create a Majordomo instance with a mocked client."""
    with patch("majordomo_llm.providers.majordomo.openai.AsyncOpenAI"):
        return Majordomo(model="glm-5.2", base_url=GATEWAY_URL, api_key="mdm-test-key")


class TestMajordomoConstruction:
    """Construction guards and header injection."""

    def test_requires_base_url(self):
        """Should reject construction without a gateway base_url."""
        with pytest.raises(ConfigurationError, match="requires base_url"):
            Majordomo(model="glm-5.2", api_key="mdm-test-key")

    def test_requires_api_key(self, monkeypatch):
        """Should require MAJORDOMO_API_KEY when no explicit key is given."""
        monkeypatch.delenv("MAJORDOMO_API_KEY", raising=False)
        with (
            patch("majordomo_llm.providers.majordomo.openai.AsyncOpenAI"),
            pytest.raises(ConfigurationError),
        ):
            Majordomo(model="glm-5.2", base_url=GATEWAY_URL)

    def test_reads_key_from_env(self, monkeypatch):
        """Should fall back to the MAJORDOMO_API_KEY environment variable."""
        monkeypatch.setenv("MAJORDOMO_API_KEY", "env-key")
        with patch("majordomo_llm.providers.majordomo.openai.AsyncOpenAI"):
            llm = Majordomo(model="glm-5.2", base_url=GATEWAY_URL)
        assert llm.default_headers["X-Majordomo-Key"] == "env-key"

    def test_injects_routing_and_auth_headers(self, majordomo_llm):
        """Should signal optimal routing (provider + model) and authenticate."""
        assert majordomo_llm.default_headers["x-majordomo-provider"] == "majordomo"
        assert majordomo_llm.default_headers["x-majordomo-model"] == "glm-5.2"
        assert majordomo_llm.default_headers["X-Majordomo-Key"] == "mdm-test-key"

    def test_caller_headers_win(self):
        """Caller default_headers should override auto-injected values."""
        with patch("majordomo_llm.providers.majordomo.openai.AsyncOpenAI"):
            llm = Majordomo(
                model="glm-5.2",
                base_url=GATEWAY_URL,
                api_key="mdm-test-key",
                default_headers={"X-Majordomo-Key": "caller-key", "X-Majordomo-Feature": "demo"},
            )
        assert llm.default_headers["X-Majordomo-Key"] == "caller-key"
        assert llm.default_headers["X-Majordomo-Feature"] == "demo"
        assert llm.default_headers["x-majordomo-provider"] == "majordomo"


class TestMajordomoGetResponse:
    """Text responses and routed-pricing behaviour."""

    async def test_returns_content_and_routed_identity(self, majordomo_llm):
        """Should surface content and the gateway-selected provider/model."""
        raw = _make_raw(
            content="Routed hello!",
            prompt_tokens=20,
            completion_tokens=8,
            routed_provider=FIREWORKS_GLM[0],
            routed_model=FIREWORKS_GLM[1],
        )
        majordomo_llm.client.chat.completions.with_raw_response.create = AsyncMock(
            return_value=raw
        )

        response = await majordomo_llm.get_response("Say hello")

        assert response.content == "Routed hello!"
        assert response.input_tokens == 20
        assert response.output_tokens == 8
        assert response.routed_provider == "fireworks"
        assert response.routed_model == "accounts/fireworks/models/glm-5p2"

    async def test_prices_from_routed_pair_with_cache(self, majordomo_llm):
        """Cost should use the routed backend's rates, including its cache tier."""
        raw = _make_raw(
            content="hi",
            prompt_tokens=1000,
            completion_tokens=200,
            cached_tokens=500,
            routed_provider=FIREWORKS_GLM[0],
            routed_model=FIREWORKS_GLM[1],
        )
        majordomo_llm.client.chat.completions.with_raw_response.create = AsyncMock(
            return_value=raw
        )

        response = await majordomo_llm.get_response("hi")

        # Fireworks GLM-5.2: input 1.4, output 4.4, cached-read 0.14 (subset).
        expected_input = (500 * 1.4 + 500 * 0.14) / TOKENS_PER_MILLION
        expected_output = (200 * 4.4) / TOKENS_PER_MILLION
        assert response.input_cost == pytest.approx(expected_input)
        assert response.output_cost == pytest.approx(expected_output)
        assert response.total_cost == pytest.approx(expected_input + expected_output)

    async def test_same_usage_priced_differently_by_route(self, majordomo_llm):
        """Identical usage should cost differently depending on the routed backend."""

        async def price_for(routed_provider, routed_model):
            raw = _make_raw(
                content="x",
                prompt_tokens=1000,
                completion_tokens=200,
                cached_tokens=500,
                routed_provider=routed_provider,
                routed_model=routed_model,
            )
            majordomo_llm.client.chat.completions.with_raw_response.create = AsyncMock(
                return_value=raw
            )
            return (await majordomo_llm.get_response("x")).total_cost

        fireworks_cost = await price_for(*FIREWORKS_GLM)  # cached-read 0.14
        together_cost = await price_for(*TOGETHER_GLM)  # cached-read 0.26

        assert together_cost > fireworks_cost

    async def test_missing_routing_headers_degrades_to_zero(self, majordomo_llm):
        """Absent routing headers should yield zero cost, not a crash."""
        raw = _make_raw(content="hi", prompt_tokens=10, completion_tokens=5)
        majordomo_llm.client.chat.completions.with_raw_response.create = AsyncMock(
            return_value=raw
        )

        response = await majordomo_llm.get_response("hi")

        assert response.total_cost == 0.0
        assert response.routed_provider is None
        assert response.routed_model is None
        assert response.input_tokens == 10

    async def test_unknown_routed_pair_degrades_to_zero(self, majordomo_llm):
        """An unconfigured routed pair should yield zero cost, not a crash."""
        raw = _make_raw(
            content="hi",
            prompt_tokens=10,
            completion_tokens=5,
            routed_provider="fireworks",
            routed_model="accounts/fireworks/models/does-not-exist",
        )
        majordomo_llm.client.chat.completions.with_raw_response.create = AsyncMock(
            return_value=raw
        )

        response = await majordomo_llm.get_response("hi")

        assert response.total_cost == 0.0
        assert response.routed_provider == "fireworks"


class TestMajordomoStructured:
    """JSON-schema / structured output."""

    async def test_structured_response_prices_from_route(self, majordomo_llm):
        """Structured output should parse and price from the routed pair."""
        raw = _make_raw(
            content='{"name": "France", "capital": "Paris", "population": 67000000}',
            prompt_tokens=50,
            completion_tokens=30,
            routed_provider=TOGETHER_GLM[0],
            routed_model=TOGETHER_GLM[1],
        )
        majordomo_llm.client.chat.completions.with_raw_response.create = AsyncMock(
            return_value=raw
        )

        response = await majordomo_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        assert isinstance(response.content, CountryInfo)
        assert response.content.capital == "Paris"
        # Together GLM-5.2: input 1.4, output 4.4 (no cached tokens here).
        expected = (50 * 1.4 + 30 * 4.4) / TOKENS_PER_MILLION
        assert response.total_cost == pytest.approx(expected)


class TestMajordomoStreaming:
    """Streaming responses price the final usage from the routed pair."""

    async def test_stream_prices_from_route(self, majordomo_llm):
        """Streamed usage should be priced with the routed backend's rates."""
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"
        chunk2.usage = None

        final_chunk = MagicMock()
        final_chunk.choices = []
        final_chunk.usage.prompt_tokens = 1000
        final_chunk.usage.completion_tokens = 200
        final_chunk.usage.prompt_tokens_details = None

        async def stream():
            yield chunk1
            yield chunk2
            yield final_chunk

        raw = MagicMock()
        raw.headers = {
            "X-Majordomo-Routed-Provider": FIREWORKS_GLM[0],
            "X-Majordomo-Routed-Model": FIREWORKS_GLM[1],
        }
        raw.parse = MagicMock(return_value=stream())
        majordomo_llm.client.chat.completions.with_raw_response.create = AsyncMock(
            return_value=raw
        )

        result = await majordomo_llm.get_response_stream("Say hello")
        collected = [chunk async for chunk in result]

        assert "".join(collected) == "Hello world"
        assert result.usage is not None
        # Fireworks GLM-5.2 rates, no cached tokens.
        expected = (1000 * 1.4 + 200 * 4.4) / TOKENS_PER_MILLION
        assert result.usage.total_cost == pytest.approx(expected)
