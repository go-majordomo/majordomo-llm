# Structured Outputs

## The Provider Fragmentation Problem

LLM providers implement structured outputs differently:

- **OpenAI** uses `response_format` with JSON schemas
- **Anthropic** uses tool calling to enforce structure
- **Gemini** uses `response_schema` in generation config
- **Others** rely on prompt engineering with varying reliability

This fragmentation creates problems:

1. **Vendor lock-in**: Code written for one provider's structured output API won't work with another
2. **Inconsistent validation**: Some providers validate server-side, others don't
3. **Schema translation**: Each provider has its own schema format and quirks
4. **Migration overhead**: Switching providers requires rewriting extraction logic

## One Method, All Providers

majordomo-llm provides a single method, `get_structured_json_response()`, that:

- Accepts a **Pydantic model** as the schema definition
- Returns a **validated, typed Python object** (not raw JSON)
- Works identically across **all supported providers**
- Handles provider-specific implementation details internally

## Basic Usage

```python
from pydantic import BaseModel, Field
from majordomo_llm import get_llm_instance

class ExtractedData(BaseModel):
    title: str = Field(description="Document title")
    summary: str = Field(description="Brief summary")
    keywords: list[str] = Field(description="Key topics")

# Same code works with any provider
llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")
# llm = get_llm_instance("openai", "gpt-4o")
# llm = get_llm_instance("gemini", "gemini-2.5-flash")

response = await llm.get_structured_json_response(
    response_model=ExtractedData,
    user_prompt="Extract info from: [your document text]",
)

# response.content is a validated ExtractedData instance
print(response.content.title)
print(response.content.keywords)
```

## Under the Hood

1. Pydantic model is converted to JSON schema via `model_json_schema()`
2. Schema is translated to provider-specific format (tool definition for Anthropic, response_format for OpenAI, etc.)
3. Provider response is parsed and validated by Pydantic
4. Typed object is returned with full IDE autocomplete support

## Next Steps

See the [Structured Outputs recipe](../recipes/structured-outputs.md) for more examples including enums, nested models, and constrained fields.
