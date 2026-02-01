# Basic Usage

Get responses from any LLM provider with a unified async interface.

## Simple Request

```python
import asyncio
from majordomo_llm import get_llm_instance

async def main():
    llm = get_llm_instance("openai", "gpt-4o")
    resp = await llm.get_response(
        user_prompt="What is the capital of France?",
        system_prompt="Answer concisely.",
        temperature=0.3,
    )
    print(resp.content)  # Paris
    print(f"Cost: ${resp.total_cost:.6f}")
    print(f"Tokens: {resp.input_tokens} in / {resp.output_tokens} out")

asyncio.run(main())
```

## Switching Providers

Use the same interface across all supported providers:

```python
# OpenAI
llm = get_llm_instance("openai", "gpt-4o")

# Anthropic
llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")

# Google Gemini
llm = get_llm_instance("gemini", "gemini-2.5-flash")

# DeepSeek
llm = get_llm_instance("deepseek", "deepseek-chat")

# Cohere
llm = get_llm_instance("cohere", "command-r-plus")
```

## Response Object

Every response includes usage metrics:

```python
resp = await llm.get_response("Hello!")

resp.content        # The response text
resp.input_tokens   # Tokens in the prompt
resp.output_tokens  # Tokens in the response
resp.total_cost     # Cost in USD
resp.response_time  # Time in seconds
```

## JSON Responses

Get raw JSON without Pydantic validation:

```python
resp = await llm.get_json_response(
    user_prompt="List 3 countries as JSON with name and capital fields",
)
print(resp.content)  # dict: {"countries": [...]}
```

## With Logging

Track all requests for analytics:

```python
from majordomo_llm import get_llm_instance
from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter

llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")

db = await SqliteAdapter.create("llm_logs.db")
storage = await FileStorageAdapter.create("./request_logs")
logged_llm = LoggingLLM(llm, db, storage)

resp = await logged_llm.get_response("Hello!")

await logged_llm.flush()  # Ensure logs are written
await db.close()
await storage.close()
```

Notes

- Set API keys via environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `CO_API_KEY`.
- Model costs are loaded from `llm_config.yaml`; add new models there.
- All methods are async; use `asyncio.run()` or an async context.
- See the [Cascade recipe](cascade.md) for automatic provider failover.
