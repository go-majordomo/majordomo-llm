# Core Concepts

This section explains the key capabilities of majordomo-llm and the problems they solve.

- **[Structured Outputs](structured-outputs.md)**: Get validated Pydantic models from any LLM provider using a single unified method. Each provider implements structured outputs differently—majordomo-llm abstracts these differences so your code works identically across OpenAI, Anthropic, Gemini, and others.

- **[Cost Tracking & Logging](cost-tracking.md)**: Every response includes token counts and calculated costs, with optional async logging for full request/response persistence. Track spending by API key, feature, or user with pluggable adapters for SQLite, Postgres, S3, and local storage.

- **[Cascade Failover](cascade-failover.md)**: Wrap multiple providers in priority order for automatic failover when one goes down. If your primary provider hits rate limits or experiences an outage, requests seamlessly fall back to the next provider in the chain.

## Composability

These capabilities work together. You can use structured outputs through a cascade with full logging:

```python
from majordomo_llm import LLMCascade
from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter

# Set up cascade
cascade = LLMCascade([
    ("anthropic", "claude-sonnet-4-20250514"),
    ("openai", "gpt-4o"),
])

# Add logging
db = await SqliteAdapter.create("llm_logs.db")
storage = await FileStorageAdapter.create("./request_logs")
logged = LoggingLLM(cascade, db, storage)

# Use structured outputs with failover and logging
response = await logged.get_structured_json_response(
    response_model=MyModel,
    user_prompt="Extract data from...",
)
```
