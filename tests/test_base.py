"""Tests for the base module."""

import pytest
from pydantic import BaseModel

from majordomo_llm.base import (
    LLM,
    LLMResponse,
    LLMJSONResponse,
    LLMStructuredResponse,
    Usage,
    TOKENS_PER_MILLION,
)


class TestUsage:
    """Tests for Usage dataclass."""

    def test_usage_stores_all_fields(self):
        """Should store all usage metrics."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=10,
            input_cost=0.0003,
            output_cost=0.00075,
            total_cost=0.00105,
            response_time=1.5,
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 10
        assert usage.input_cost == 0.0003
        assert usage.output_cost == 0.00075
        assert usage.total_cost == 0.00105
        assert usage.response_time == 1.5


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_includes_content_and_usage(self):
        """Should include content and inherit usage fields."""
        response = LLMResponse(
            content="Hello, world!",
            input_tokens=10,
            output_tokens=5,
            cached_tokens=0,
            input_cost=0.00003,
            output_cost=0.000075,
            total_cost=0.000105,
            response_time=0.5,
        )

        assert response.content == "Hello, world!"
        assert response.input_tokens == 10
        assert response.output_tokens == 5


class TestLLMJSONResponse:
    """Tests for LLMJSONResponse dataclass."""

    def test_content_is_dict(self):
        """Content should be a dictionary."""
        response = LLMJSONResponse(
            content={"key": "value", "number": 42},
            input_tokens=20,
            output_tokens=10,
            cached_tokens=0,
            input_cost=0.00006,
            output_cost=0.00015,
            total_cost=0.00021,
            response_time=0.8,
        )

        assert response.content == {"key": "value", "number": 42}
        assert response.content["key"] == "value"


class TestLLMStructuredResponse:
    """Tests for LLMStructuredResponse dataclass."""

    def test_content_is_pydantic_model(self):
        """Content should be a Pydantic model instance."""

        class Person(BaseModel):
            name: str
            age: int

        person = Person(name="Alice", age=30)
        response = LLMStructuredResponse(
            content=person,
            input_tokens=30,
            output_tokens=15,
            cached_tokens=5,
            input_cost=0.00009,
            output_cost=0.000225,
            total_cost=0.000315,
            response_time=1.0,
        )

        assert response.content.name == "Alice"
        assert response.content.age == 30


class TestLLMCostCalculation:
    """Tests for LLM._calculate_costs method."""

    class ConcreteLLM(LLM):
        """Concrete implementation for testing abstract base class."""

        async def get_response(self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0):
            raise NotImplementedError()

    def test_calculates_costs_correctly(self):
        """Should calculate costs based on tokens and rates."""
        llm = self.ConcreteLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,  # $3 per million tokens
            output_cost=15.0,  # $15 per million tokens
        )

        input_cost, output_cost, total_cost = llm._calculate_costs(
            input_tokens=1000,
            output_tokens=500,
        )

        expected_input = 1000 * 3.0 / TOKENS_PER_MILLION
        expected_output = 500 * 15.0 / TOKENS_PER_MILLION

        assert input_cost == expected_input
        assert output_cost == expected_output
        assert total_cost == expected_input + expected_output

    def test_handles_zero_tokens(self):
        """Should handle zero tokens gracefully."""
        llm = self.ConcreteLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,
            output_cost=15.0,
        )

        input_cost, output_cost, total_cost = llm._calculate_costs(
            input_tokens=0,
            output_tokens=0,
        )

        assert input_cost == 0.0
        assert output_cost == 0.0
        assert total_cost == 0.0

    def test_handles_large_token_counts(self):
        """Should handle large token counts correctly."""
        llm = self.ConcreteLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,
            output_cost=15.0,
        )

        input_cost, output_cost, total_cost = llm._calculate_costs(
            input_tokens=1_000_000,  # 1 million tokens
            output_tokens=1_000_000,
        )

        assert input_cost == 3.0  # Exactly $3
        assert output_cost == 15.0  # Exactly $15
        assert total_cost == 18.0


class TestLLMFullModelName:
    """Tests for LLM.get_full_model_name method."""

    class ConcreteLLM(LLM):
        """Concrete implementation for testing."""

        async def get_response(self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0):
            raise NotImplementedError()

    def test_returns_provider_colon_model(self):
        """Should return 'provider:model' format."""
        llm = self.ConcreteLLM(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            input_cost=3.0,
            output_cost=15.0,
        )

        assert llm.get_full_model_name() == "anthropic:claude-sonnet-4-20250514"
