# Cost Tracking & Logging

## The Visibility Problem

LLM API costs can escalate quickly and unpredictably:

- **No visibility**: Without tracking, you discover cost issues after the invoice arrives
- **Attribution difficulty**: Hard to know which features, users, or prompts drive costs
- **Debugging challenges**: Can't replay or analyze requests without logs
- **Compliance needs**: Some use cases require audit trails of AI interactions

## Two Layers of Cost Awareness

majordomo-llm provides two layers of cost awareness:

1. **Built-in usage metrics**: Every response includes token counts and calculated costs
2. **Optional logging subsystem**: Async, non-blocking persistence of all requests with full request/response bodies

The logging system uses pluggable adapters:

| Component | Local Development | Production |
|-----------|-------------------|------------|
| Metadata DB | `SqliteAdapter` | `PostgresAdapter`, `MySQLAdapter` |
| Body Storage | `FileStorageAdapter` | `S3Adapter` |

## Basic Cost Tracking

No logging required—every response includes usage metrics:

```python
response = await llm.get_response("Hello!")

print(f"Input tokens: {response.input_tokens}")
print(f"Output tokens: {response.output_tokens}")
print(f"Total cost: ${response.total_cost:.6f}")
print(f"Response time: {response.response_time:.2f}s")
```

Costs are calculated using per-model rates from `llm_config.yaml`:

```yaml
gpt-4o:
  input_cost_per_million: 2.5
  output_cost_per_million: 10.0
```

## Full Request Logging

For persistence and analytics:

```python
from majordomo_llm import get_llm_instance
from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter

# Create adapters
db = await SqliteAdapter.create("llm_logs.db")
storage = await FileStorageAdapter.create("./request_logs")

# Wrap any LLM instance
llm = get_llm_instance("openai", "gpt-4o", api_key_alias="prod-key-1")
logged_llm = LoggingLLM(llm, db, storage)

# Use normally - logging happens async in background
response = await logged_llm.get_response("Hello!")

# Ensure logs are written before shutdown
await logged_llm.flush()
```

## What Gets Logged

- Timestamp, provider, model
- Token counts and costs
- Response time
- API key hash and alias (for attribution)
- Full request/response bodies (stored separately in S3/filesystem)

## Next Steps

See the [Cost Tracking & Logging recipe](../recipes/logging.md) for production configurations with Postgres and S3.
