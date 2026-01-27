#!/usr/bin/env python3
"""Demo script showcasing structured JSON responses with Pydantic models.

This script demonstrates:
- Using get_structured_json_response() to get validated Pydantic models
- Different schema patterns (simple, nested, lists, enums)
- Comparing structured output support across providers
- Logging all requests to SQLite with cost summary

Prerequisites:
    1. Install dependencies: uv sync --all-extras
    2. Set at least one API key:
       - OPENAI_API_KEY
       - ANTHROPIC_API_KEY
       - GEMINI_API_KEY
       - DEEPSEEK_API_KEY
       - CO_API_KEY

Usage:
    uv run python examples/structured_response_demo.py
"""

import asyncio
from enum import Enum

from pydantic import BaseModel, Field
from shared import (
    EXAMPLES_DIR,
    clear_database,
    get_available_providers,
    print_summary,
)

from majordomo_llm import get_llm_instance
from majordomo_llm.logging import FileStorageAdapter, LoggingLLM, SqliteAdapter

# Output paths
DB_PATH = EXAMPLES_DIR / "structured_logs.db"
STORAGE_DIR = EXAMPLES_DIR / "structured_request_logs"


# =============================================================================
# Pydantic Models for Structured Outputs
# =============================================================================


class Sentiment(str, Enum):
    """Sentiment classification."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentAnalysis(BaseModel):
    """Result of sentiment analysis."""

    sentiment: Sentiment = Field(description="The overall sentiment")
    confidence: float = Field(ge=0, le=1, description="Confidence score from 0 to 1")
    reasoning: str = Field(description="Brief explanation for the sentiment")


class Entity(BaseModel):
    """A named entity extracted from text."""

    name: str = Field(description="The entity name")
    type: str = Field(description="Entity type (person, organization, location, etc.)")
    relevance: float = Field(ge=0, le=1, description="Relevance score from 0 to 1")


class TextAnalysis(BaseModel):
    """Comprehensive text analysis with nested entities."""

    summary: str = Field(description="One-sentence summary of the text")
    entities: list[Entity] = Field(description="Named entities found in the text")
    topics: list[str] = Field(description="Main topics discussed")
    word_count: int = Field(description="Approximate word count")


class CodeReview(BaseModel):
    """Code review feedback."""

    quality_score: int = Field(ge=1, le=10, description="Code quality from 1-10")
    issues: list[str] = Field(description="List of issues found")
    suggestions: list[str] = Field(description="Improvement suggestions")
    is_production_ready: bool = Field(description="Whether code is production-ready")


class ProductRecommendation(BaseModel):
    """A product recommendation."""

    name: str = Field(description="Product name")
    price_range: str = Field(description="Price range (e.g., '$50-100')")
    pros: list[str] = Field(description="Advantages of this product")
    cons: list[str] = Field(description="Disadvantages of this product")
    rating: float = Field(ge=0, le=5, description="Rating from 0 to 5")


class ProductRecommendations(BaseModel):
    """List of product recommendations."""

    query: str = Field(description="The original search query")
    recommendations: list[ProductRecommendation] = Field(
        description="List of recommended products", min_length=1, max_length=5
    )
    best_value: str = Field(description="Name of the best value option")


# =============================================================================
# Demo Functions
# =============================================================================


async def demo_sentiment_analysis(
    logged_llm: LoggingLLM, provider: str, model: str
) -> None:
    """Demo: Simple enum-based structured output."""
    print(f"\n  [{provider}/{model}]")

    text = """
    I just received my order and I'm absolutely thrilled! The packaging was
    beautiful, the product exceeded my expectations, and it arrived two days
    early. Best purchase I've made this year!
    """

    try:
        response = await logged_llm.get_structured_json_response(
            response_model=SentimentAnalysis,
            user_prompt=f"Analyze the sentiment of this text:\n\n{text}",
            system_prompt="You are a sentiment analysis expert.",
        )

        result: SentimentAnalysis = response.content
        print(f"    Sentiment: {result.sentiment.value} ({result.confidence:.0%} confidence)")
        print(f"    Reasoning: {result.reasoning[:80]}...")
        print(f"    Cost: ${response.total_cost:.6f} | "
              f"Tokens: {response.input_tokens}+{response.output_tokens}")
    except Exception as e:
        print(f"    Error: {e}")


async def demo_text_analysis(
    logged_llm: LoggingLLM, provider: str, model: str
) -> None:
    """Demo: Nested model with lists."""
    print(f"\n  [{provider}/{model}]")

    text = """
    Apple CEO Tim Cook announced today that the company will invest $1 billion
    in a new artificial intelligence research center in Austin, Texas. The
    facility, expected to open in 2026, will focus on developing next-generation
    machine learning technologies. Cook stated that this investment reflects
    Apple's commitment to American innovation and job creation, with plans to
    hire over 3,000 engineers and researchers.
    """

    try:
        response = await logged_llm.get_structured_json_response(
            response_model=TextAnalysis,
            user_prompt=f"Analyze this news article:\n\n{text}",
            system_prompt="You are a text analysis expert. Extract key information.",
        )

        result: TextAnalysis = response.content
        print(f"    Summary: {result.summary[:70]}...")
        print(f"    Topics: {', '.join(result.topics[:3])}")
        entities_str = ", ".join(f"{e.name} ({e.type})" for e in result.entities[:3])
        print(f"    Entities: {entities_str}")
        print(f"    Cost: ${response.total_cost:.6f} | "
              f"Tokens: {response.input_tokens}+{response.output_tokens}")
    except Exception as e:
        print(f"    Error: {e}")


async def demo_code_review(
    logged_llm: LoggingLLM, provider: str, model: str
) -> None:
    """Demo: Boolean and constrained integer fields."""
    print(f"\n  [{provider}/{model}]")

    code = '''
def calculate_total(items):
    total = 0
    for i in range(len(items)):
        total = total + items[i]["price"] * items[i]["quantity"]
    return total
'''

    try:
        response = await logged_llm.get_structured_json_response(
            response_model=CodeReview,
            user_prompt=f"Review this Python code:\n```python\n{code}\n```",
            system_prompt="You are an expert Python code reviewer. Be constructive but thorough.",
        )

        result: CodeReview = response.content
        ready = "Yes" if result.is_production_ready else "No"
        print(f"    Quality: {result.quality_score}/10 | Production Ready: {ready}")
        print(f"    Issues: {len(result.issues)} found")
        if result.issues:
            print(f"      - {result.issues[0][:60]}...")
        print(f"    Cost: ${response.total_cost:.6f} | "
              f"Tokens: {response.input_tokens}+{response.output_tokens}")
    except Exception as e:
        print(f"    Error: {e}")


async def demo_product_recommendations(
    logged_llm: LoggingLLM, provider: str, model: str
) -> None:
    """Demo: Complex nested lists with multiple constraints."""
    print(f"\n  [{provider}/{model}]")

    query = "wireless noise-cancelling headphones under $300"

    try:
        response = await logged_llm.get_structured_json_response(
            response_model=ProductRecommendations,
            user_prompt=f"Recommend products for: {query}",
            system_prompt=(
                "You are a product recommendation expert. Provide realistic, "
                "helpful recommendations based on current market offerings."
            ),
        )

        result: ProductRecommendations = response.content
        print(f"    Best Value: {result.best_value}")
        print(f"    Recommendations: {len(result.recommendations)} products")
        for rec in result.recommendations[:2]:
            print(f"      - {rec.name} ({rec.price_range}) - {rec.rating}/5")
        print(f"    Cost: ${response.total_cost:.6f} | "
              f"Tokens: {response.input_tokens}+{response.output_tokens}")
    except Exception as e:
        print(f"    Error: {e}")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run all structured response demos across all available providers."""
    print("=" * 100)
    print("majordomo-llm: Structured Response Demo (All Providers)")
    print("=" * 100)

    # Get all available providers
    available_providers = get_available_providers()
    if not available_providers:
        print("No API keys found. Please set at least one of:")
        print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY,")
        print("  DEEPSEEK_API_KEY, or CO_API_KEY")
        return

    print(f"\nAvailable providers: {', '.join(p[0] for p in available_providers)}")

    # Clear database before running
    print("\nClearing previous logs...")
    await clear_database(DB_PATH, STORAGE_DIR)

    print(f"Logging to: {DB_PATH}")
    print(f"Request/response bodies stored in: {STORAGE_DIR}\n")

    # Initialize logging
    db = await SqliteAdapter.create(str(DB_PATH))
    storage = await FileStorageAdapter.create(STORAGE_DIR)
    logged_llms: list[LoggingLLM] = []

    try:
        # Demo 1: Sentiment Analysis
        print("-" * 100)
        print("Demo 1: Sentiment Analysis (Enum fields)")
        print("-" * 100)
        for provider, model in available_providers:
            llm = get_llm_instance(provider, model)
            logged_llm = LoggingLLM(llm, db, storage)
            logged_llms.append(logged_llm)
            await demo_sentiment_analysis(logged_llm, provider, model)

        # Demo 2: Text Analysis
        print("\n" + "-" * 100)
        print("Demo 2: Text Analysis (Nested models with lists)")
        print("-" * 100)
        for provider, model in available_providers:
            llm = get_llm_instance(provider, model)
            logged_llm = LoggingLLM(llm, db, storage)
            logged_llms.append(logged_llm)
            await demo_text_analysis(logged_llm, provider, model)

        # Demo 3: Code Review
        print("\n" + "-" * 100)
        print("Demo 3: Code Review (Constrained integers, booleans)")
        print("-" * 100)
        for provider, model in available_providers:
            llm = get_llm_instance(provider, model)
            logged_llm = LoggingLLM(llm, db, storage)
            logged_llms.append(logged_llm)
            await demo_code_review(logged_llm, provider, model)

        # Demo 4: Product Recommendations
        print("\n" + "-" * 100)
        print("Demo 4: Product Recommendations (Complex nested lists)")
        print("-" * 100)
        for provider, model in available_providers:
            llm = get_llm_instance(provider, model)
            logged_llm = LoggingLLM(llm, db, storage)
            logged_llms.append(logged_llm)
            await demo_product_recommendations(logged_llm, provider, model)

        # Flush all pending logging tasks
        for logged_llm in logged_llms:
            await logged_llm.flush()

        # Print cost summary
        await print_summary(DB_PATH)

    finally:
        await db.close()
        await storage.close()

    print(f"\nDone! Check {STORAGE_DIR} for request/response JSON files.")


if __name__ == "__main__":
    asyncio.run(main())
