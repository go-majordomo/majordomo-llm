"""Tests for the Gemini provider."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from majordomo_llm.base import TOKENS_PER_MILLION
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers import Gemini


class CountryInfo(BaseModel):
    """Test model for structured responses."""

    name: str
    capital: str
    population: int


COUNTRY_SCHEMA = CountryInfo.model_json_schema()


class TestGeminiGetResponse:
    """Tests for Gemini.get_response method."""

    @pytest.fixture
    def gemini_llm(self):
        """Create Gemini instance with mocked client."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                api_key="test-key",
            )
            return llm

    async def test_returns_text_content(self, gemini_llm, mock_gemini_text_response):
        """Should extract text content from response."""
        gemini_llm.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_text_response
        )

        response = await gemini_llm.get_response("Say hello")

        assert response.content == "Gemini says hello!"

    async def test_returns_correct_token_counts(self, gemini_llm, mock_gemini_text_response):
        """Should return correct token counts."""
        gemini_llm.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_text_response
        )

        response = await gemini_llm.get_response("Test prompt")

        assert response.input_tokens == 15
        assert response.output_tokens == 5
        assert response.cached_tokens == 0  # Gemini doesn't report cached tokens

    async def test_calculates_costs_correctly(self, gemini_llm, mock_gemini_text_response):
        """Should calculate costs based on token counts and rates."""
        gemini_llm.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_text_response
        )

        response = await gemini_llm.get_response("Test prompt")

        expected_input_cost = 15 * 0.30 / TOKENS_PER_MILLION
        expected_output_cost = 5 * 2.50 / TOKENS_PER_MILLION

        assert response.input_cost == expected_input_cost
        assert response.output_cost == expected_output_cost
        assert response.total_cost == expected_input_cost + expected_output_cost

    async def test_passes_config_parameters(self, gemini_llm, mock_gemini_text_response):
        """Should pass temperature and top_p in config."""
        gemini_llm.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_text_response
        )

        await gemini_llm.get_response(
            "Test prompt",
            system_prompt="Be helpful",
            temperature=0.5,
            top_p=0.8,
        )

        call_kwargs = gemini_llm.client.aio.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.temperature == 0.5
        assert config.top_p == 0.8
        assert config.system_instruction == "Be helpful"


class TestGeminiGetJSONResponse:
    """Tests for Gemini.get_json_response method."""

    @pytest.fixture
    def gemini_llm(self):
        """Create Gemini instance with mocked client."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                api_key="test-key",
            )
            return llm

    async def test_parses_json_response(self, gemini_llm, mock_gemini_json_response):
        """Should parse JSON from response text."""
        gemini_llm.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_json_response
        )

        response = await gemini_llm.get_json_response("Return JSON")

        assert response.content == {"status": "success", "value": 123}

    async def test_strips_markdown_fences(self, gemini_llm):
        """Should strip markdown code fences from JSON."""
        mock_response = MagicMock()
        mock_response.text = '```json\n{"key": "value"}\n```'
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 8

        gemini_llm.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        response = await gemini_llm.get_json_response("Return JSON")

        assert response.content == {"key": "value"}


class TestGeminiStructuredResponse:
    """Tests for Gemini structured response methods."""

    @pytest.fixture
    def gemini_llm(self):
        """Create Gemini instance with mocked client."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                api_key="test-key",
            )
            return llm

    async def test_passes_response_schema(self, gemini_llm):
        """Should pass JSON schema as response_schema in config."""
        mock_response = MagicMock()
        mock_response.text = '{"name": "Germany", "capital": "Berlin", "population": 83000000}'
        mock_response.usage_metadata.prompt_token_count = 30
        mock_response.usage_metadata.candidates_token_count = 20

        gemini_llm.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await gemini_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about Germany",
        )

        call_kwargs = gemini_llm.client.aio.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.response_schema is not None
        assert config.response_mime_type == "application/json"

    async def test_inlines_nested_model_refs(self, gemini_llm):
        """Nested models emit $defs/$ref, which Gemini cannot resolve; inline them."""

        class Address(BaseModel):
            city: str
            country: str

        class Company(BaseModel):
            name: str
            headquarters: Address

        mock_response = MagicMock()
        mock_response.text = (
            '{"name": "Acme", "headquarters": {"city": "Berlin", "country": "Germany"}}'
        )
        mock_response.usage_metadata.prompt_token_count = 30
        mock_response.usage_metadata.candidates_token_count = 20

        gemini_llm.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await gemini_llm.get_structured_json_response(
            response_model=Company,
            user_prompt="Describe Acme",
        )

        config = gemini_llm.client.aio.models.generate_content.call_args.kwargs["config"]
        schema_str = json.dumps(config.response_schema)
        assert "$defs" not in schema_str
        assert "$ref" not in schema_str
        headquarters = config.response_schema["properties"]["headquarters"]
        assert headquarters["type"] == "object"
        assert set(headquarters["properties"]) == {"city", "country"}

    async def test_json_schema_response_returns_canonical_json(self, gemini_llm):
        """Should pass raw schema and return canonical JSON string."""
        mock_response = MagicMock()
        mock_response.text = '{"name": "Germany", "capital": "Berlin", "population": 83000000}'
        mock_response.usage_metadata.prompt_token_count = 30
        mock_response.usage_metadata.candidates_token_count = 20

        gemini_llm.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        response = await gemini_llm.get_json_schema_response(
            user_prompt="Tell me about Germany",
            response_schema=COUNTRY_SCHEMA,
            schema_name="CountryInfo",
        )

        call_kwargs = gemini_llm.client.aio.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.response_schema == COUNTRY_SCHEMA
        assert response.content == '{"capital":"Berlin","name":"Germany","population":83000000}'

    async def test_returns_validated_pydantic_model(self, gemini_llm):
        """Should return a validated Pydantic model instance."""
        mock_response = MagicMock()
        mock_response.text = '{"name": "Germany", "capital": "Berlin", "population": 83000000}'
        mock_response.usage_metadata.prompt_token_count = 30
        mock_response.usage_metadata.candidates_token_count = 20

        gemini_llm.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        response = await gemini_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about Germany",
        )

        assert isinstance(response.content, CountryInfo)
        assert response.content.name == "Germany"
        assert response.content.capital == "Berlin"


class TestGeminiInit:
    """Tests for Gemini initialization."""

    def test_raises_configuration_error_without_api_key(self):
        """Should raise ConfigurationError when no API key is provided."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                Gemini(
                    model="gemini-2.5-flash",
                    input_cost=0.30,
                    output_cost=2.50,
                )

            assert "GEMINI_API_KEY" in str(exc_info.value)

    def test_sets_provider_name(self):
        """Should set provider to 'gemini'."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                api_key="test-key",
            )

            assert llm.provider == "gemini"

    def test_stores_model_and_costs(self):
        """Should store model name and cost configuration."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                api_key="test-key",
            )

            assert llm.model == "gemini-2.5-flash"
            assert llm.input_cost == 0.30
            assert llm.output_cost == 2.50

    def test_always_supports_temperature(self):
        """Should always support temperature/top_p."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                api_key="test-key",
            )

            assert llm.supports_temperature_top_p is True


class TestGeminiGetResponseStream:
    """Tests for Gemini.get_response_stream method."""

    @pytest.fixture
    def gemini_llm(self):
        """Create Gemini instance with mocked client."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                api_key="test-key",
            )
            return llm

    async def test_yields_text_chunks(self, gemini_llm, mock_gemini_stream_chunks):
        """Should yield text chunks from stream."""
        gemini_llm.client.aio.models.generate_content_stream = AsyncMock(
            return_value=mock_gemini_stream_chunks
        )

        stream = await gemini_llm.get_response_stream("Hello")
        chunks = [chunk async for chunk in stream]

        assert chunks == ["Hello", " world"]

    async def test_usage_populated_after_iteration(self, gemini_llm, mock_gemini_stream_chunks):
        """Should populate usage after stream is consumed."""
        gemini_llm.client.aio.models.generate_content_stream = AsyncMock(
            return_value=mock_gemini_stream_chunks
        )

        stream = await gemini_llm.get_response_stream("Hello")
        async for _ in stream:
            pass

        assert stream.usage is not None
        assert stream.usage.input_tokens == 15
        assert stream.usage.output_tokens == 5

    async def test_calculates_costs_correctly(self, gemini_llm, mock_gemini_stream_chunks):
        """Should calculate costs from stream usage."""
        gemini_llm.client.aio.models.generate_content_stream = AsyncMock(
            return_value=mock_gemini_stream_chunks
        )

        stream = await gemini_llm.get_response_stream("Hello")
        async for _ in stream:
            pass

        expected_input_cost = 15 * 0.30 / TOKENS_PER_MILLION
        expected_output_cost = 5 * 2.50 / TOKENS_PER_MILLION
        assert stream.usage.input_cost == expected_input_cost
        assert stream.usage.output_cost == expected_output_cost

    async def test_collect_returns_llm_response(self, gemini_llm, mock_gemini_stream_chunks):
        """Should return LLMResponse from collect()."""
        gemini_llm.client.aio.models.generate_content_stream = AsyncMock(
            return_value=mock_gemini_stream_chunks
        )

        stream = await gemini_llm.get_response_stream("Hello")
        response = await stream.collect()

        assert response.content == "Hello world"
        assert response.input_tokens == 15


class TestGeminiWebSearch:
    """Tests for Gemini web search wiring."""

    @pytest.fixture
    def gemini_llm_web(self):
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            return Gemini(
                model="gemini-2.5-flash",
                input_cost=0.30,
                output_cost=2.50,
                use_web_search=True,
                api_key="test-key",
            )

    async def test_google_search_tool_in_config(self, gemini_llm_web, mock_gemini_text_response):
        gemini_llm_web.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_text_response
        )

        await gemini_llm_web.get_response("Latest news?")

        config = gemini_llm_web.client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.tools is not None
        assert len(config.tools) == 1
        assert config.tools[0].google_search is not None

    async def test_grounded_query_cost_added(self, gemini_llm_web, mock_gemini_text_response):
        candidate = MagicMock()
        candidate.grounding_metadata = MagicMock()
        mock_gemini_text_response.candidates = [candidate]
        gemini_llm_web.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_text_response
        )

        response = await gemini_llm_web.get_response("Latest news?")

        assert response.tool_use_cost == pytest.approx(0.035)
        expected_input_cost = 15 * 0.30 / TOKENS_PER_MILLION
        expected_output_cost = 5 * 2.50 / TOKENS_PER_MILLION
        assert response.total_cost == pytest.approx(
            expected_input_cost + expected_output_cost + 0.035
        )

    async def test_structured_call_raises_when_grounding_enabled(
        self, gemini_llm_web, mock_gemini_json_response
    ):
        gemini_llm_web.client.aio.models.generate_content = AsyncMock(
            return_value=mock_gemini_json_response
        )

        with pytest.raises(ConfigurationError) as exc_info:
            await gemini_llm_web.get_json_schema_response(
                "Return status",
                response_schema={"type": "object", "properties": {"status": {"type": "string"}}},
            )

        assert "grounded web search" in str(exc_info.value)

    async def test_structured_call_allows_grounding_for_gemini_3(self):
        """Gemini 3 series models may combine grounding with a response schema."""
        with patch("majordomo_llm.providers.gemini.genai.Client"):
            llm = Gemini(
                model="gemini-3.6-flash",
                input_cost=0.30,
                output_cost=2.50,
                use_web_search=True,
                api_key="test-key",
            )

        mock_response = MagicMock()
        mock_response.text = '{"status": "ok"}'
        mock_response.usage_metadata.prompt_token_count = 30
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.candidates = []
        llm.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        response = await llm.get_json_schema_response(
            "Return status",
            response_schema={"type": "object", "properties": {"status": {"type": "string"}}},
        )

        config = llm.client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.tools is not None
        assert config.tools[0].google_search is not None
        assert config.response_mime_type == "application/json"
        assert response.content == '{"status":"ok"}'
