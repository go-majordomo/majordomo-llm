# Structured Outputs with Pydantic

Validate responses to Pydantic models using `get_structured_json_response`.

## Simple Model

```python
from pydantic import BaseModel, Field
from majordomo_llm import get_llm_instance

class Country(BaseModel):
    name: str
    capital: str
    population: int

llm = get_llm_instance("openai", "gpt-4o")
resp = await llm.get_structured_json_response(
    response_model=Country,
    user_prompt="Return info about Japan as JSON",
)
country: Country = resp.content
print(country.capital)  # Tokyo
```

## Enum Fields

Use enums for classification tasks:

```python
from enum import Enum
from pydantic import BaseModel, Field

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class SentimentAnalysis(BaseModel):
    sentiment: Sentiment = Field(description="The overall sentiment")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    reasoning: str = Field(description="Brief explanation")

resp = await llm.get_structured_json_response(
    response_model=SentimentAnalysis,
    user_prompt="Analyze the sentiment: 'Best purchase I've made this year!'",
    system_prompt="You are a sentiment analysis expert.",
)
result: SentimentAnalysis = resp.content
print(f"{result.sentiment.value} ({result.confidence:.0%})")
```

## Nested Models with Lists

Extract structured data with nested entities:

```python
class Entity(BaseModel):
    name: str = Field(description="The entity name")
    type: str = Field(description="Entity type (person, org, location)")
    relevance: float = Field(ge=0, le=1, description="Relevance score")

class TextAnalysis(BaseModel):
    summary: str = Field(description="One-sentence summary")
    entities: list[Entity] = Field(description="Named entities found")
    topics: list[str] = Field(description="Main topics discussed")

resp = await llm.get_structured_json_response(
    response_model=TextAnalysis,
    user_prompt="Analyze: Apple CEO Tim Cook announced a $1B AI center in Austin.",
    system_prompt="Extract key information from the text.",
)
for entity in resp.content.entities:
    print(f"{entity.name} ({entity.type})")
```

## Constrained Fields

Use Field constraints for validation:

```python
class CodeReview(BaseModel):
    quality_score: int = Field(ge=1, le=10, description="Quality from 1-10")
    issues: list[str] = Field(description="Issues found")
    suggestions: list[str] = Field(description="Improvement suggestions")
    is_production_ready: bool = Field(description="Ready for production")

resp = await llm.get_structured_json_response(
    response_model=CodeReview,
    user_prompt="Review this code:\n```python\ndef add(a, b): return a + b\n```",
    system_prompt="You are an expert code reviewer.",
)
print(f"Quality: {resp.content.quality_score}/10")
```

## Complex Nested Lists

For multi-item responses with constraints:

```python
class ProductRecommendation(BaseModel):
    name: str = Field(description="Product name")
    price_range: str = Field(description="Price range")
    pros: list[str] = Field(description="Advantages")
    cons: list[str] = Field(description="Disadvantages")
    rating: float = Field(ge=0, le=5, description="Rating 0-5")

class ProductRecommendations(BaseModel):
    query: str = Field(description="Original search query")
    recommendations: list[ProductRecommendation] = Field(
        min_length=1, max_length=5
    )
    best_value: str = Field(description="Best value option name")

resp = await llm.get_structured_json_response(
    response_model=ProductRecommendations,
    user_prompt="Recommend wireless headphones under $300",
)
for rec in resp.content.recommendations:
    print(f"{rec.name}: {rec.rating}/5")
```

Notes

- Pydantic validates and coerces types; handle `ValidationError` for bad outputs.
- Use `Field(description=...)` to guide the LLM on expected values.
- All providers support structured outputs; Anthropic uses tool calling, others use response schemas.
- Response includes usage metrics: `resp.total_cost`, `resp.input_tokens`, `resp.output_tokens`.
