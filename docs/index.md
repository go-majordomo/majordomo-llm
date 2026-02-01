# majordomo-llm

A unified async Python interface for multiple LLM providers with built-in cost tracking, automatic retries, and structured outputs.

## Why majordomo-llm?

Building with LLMs often means dealing with:

- **Different APIs for each provider** — OpenAI, Anthropic, and Gemini all have different client libraries and response formats
- **Hidden costs** — Token usage and spending are hard to track across providers
- **Fragile integrations** — When one provider goes down, your application goes down
- **Inconsistent structured outputs** — Each provider handles JSON schemas differently

majordomo-llm solves these problems with a single, consistent interface that works across all major providers.

## Quick Example

```python
import asyncio
from pydantic import BaseModel
from majordomo_llm import get_llm_instance

class Summary(BaseModel):
    title: str
    key_points: list[str]
    word_count: int

async def main():
    # Works with any provider: openai, anthropic, gemini, deepseek, cohere
    llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")

    response = await llm.get_structured_json_response(
        response_model=Summary,
        user_prompt="Summarize the benefits of async programming in Python",
    )

    print(response.content.title)
    print(response.content.key_points)
    print(f"Cost: ${response.total_cost:.6f}")

asyncio.run(main())
```

## Key Features

### Unified Provider Interface

Write once, run on any provider. Switch between OpenAI, Anthropic, Gemini, DeepSeek, and Cohere with a single line change.

```python
llm = get_llm_instance("openai", "gpt-4o")
llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")
llm = get_llm_instance("gemini", "gemini-2.5-flash")
```

### Structured Outputs with Pydantic

Get validated, typed Python objects instead of raw JSON. Provider-specific implementation details are handled internally.

```python
response = await llm.get_structured_json_response(
    response_model=MyPydanticModel,
    user_prompt="Extract data from this text...",
)
result: MyPydanticModel = response.content  # Fully typed
```

### Built-in Cost Tracking

Every response includes token counts and calculated costs. No external tracking needed.

```python
print(f"Tokens: {response.input_tokens} in / {response.output_tokens} out")
print(f"Cost: ${response.total_cost:.6f}")
```

### Cascade Failover

Automatically fall back to alternative providers when one fails.

```python
from majordomo_llm import LLMCascade

cascade = LLMCascade([
    ("anthropic", "claude-sonnet-4-20250514"),
    ("openai", "gpt-4o"),
    ("gemini", "gemini-2.5-flash"),
])
response = await cascade.get_response("Hello!")  # Tries each until one succeeds
```

### Optional Request Logging

Persist all requests for analytics, debugging, and compliance with pluggable database and storage adapters.

```python
from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter

db = await SqliteAdapter.create("logs.db")
storage = await FileStorageAdapter.create("./request_logs")
logged_llm = LoggingLLM(llm, db, storage)
```

## Supported Providers

| Provider | Recent Models |
|----------|---------------|
| OpenAI | gpt-5, gpt-5-mini, gpt-4.1, gpt-4.1-mini, gpt-4o |
| Anthropic | claude-sonnet-4.5, claude-opus-4.1, claude-sonnet-4, claude-3.5-haiku |
| Google Gemini | gemini-2.5-flash, gemini-2.0-flash |
| DeepSeek | deepseek-chat, deepseek-reasoner |
| Cohere | command-a, command-r-plus, command-r |

All providers support structured outputs. Additional models are available—see `llm_config.yaml` for the complete list with pricing.

## Next Steps

- [Getting Started](getting-started.md) — Installation and quickstart
- [Core Concepts](concepts/index.md) — Understand the key capabilities
- [Recipes](recipes/basic-usage.md) — Practical examples and patterns
- [API Reference](api/base.md) — Detailed API documentation
