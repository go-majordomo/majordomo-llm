# majordomo-llm

[![PyPI version](https://badge.fury.io/py/majordomo-llm.svg)](https://badge.fury.io/py/majordomo-llm)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-website-blue.svg)](https://superset-studio.github.io/majordomo-llm/)

A unified Python interface for multiple LLM providers with automatic cost tracking, retry logic, and structured output support.

## Features

- **Unified API** - Same interface for OpenAI, Anthropic (Claude), Google Gemini, DeepSeek, and Cohere
- **Cost Tracking** - Automatic calculation of input/output token costs per request
- **Structured Outputs** - Native support for Pydantic models as response schemas
- **Automatic Retries** - Built-in exponential backoff retry logic using tenacity
- **Automatic Fallback** - Cascade across providers with `LLMCascade` for resilience
- **Request Logging** - Optional async logging to PostgreSQL/MySQL/SQLite with S3 or local file storage for request/response bodies
- **API Key Tracking** - Log hashed API keys and optional aliases for usage attribution
- **Async First** - Fully async/await compatible for high-performance applications
- **Type Safe** - Complete type annotations and `py.typed` marker for IDE support

## Installation

```bash
pip install majordomo-llm
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add majordomo-llm
```

### Optional: Request Logging

To enable request logging to PostgreSQL, MySQL, or S3:

```bash
pip install majordomo-llm[logging]
```

## Quick Start

### Basic Text Response

```python
import asyncio
from majordomo_llm import get_llm_instance

async def main():
    # Create an LLM instance
    llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")

    # Get a response
    response = await llm.get_response(
        user_prompt="What is the capital of France?",
        system_prompt="You are a helpful geography assistant.",
    )

    print(response.content)
    print(f"Tokens: {response.input_tokens} in, {response.output_tokens} out")
    print(f"Cost: ${response.total_cost:.6f}")

asyncio.run(main())
```

### JSON Response

```python
response = await llm.get_json_response(
    user_prompt="List the top 3 largest countries by area as JSON",
    system_prompt="Respond with valid JSON only.",
)

# response.content is a parsed Python dict
for country in response.content["countries"]:
    print(country["name"])
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel

class CountryInfo(BaseModel):
    name: str
    capital: str
    population: int
    area_km2: float

response = await llm.get_structured_json_response(
    response_model=CountryInfo,
    user_prompt="Give me information about Japan",
)

# response.content is a validated CountryInfo instance
country = response.content
print(f"{country.name}: {country.capital}, pop. {country.population:,}")
```

## Configuration

### Environment Variables

Set API keys for the providers you want to use:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GEMINI_API_KEY="..."

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."

# Cohere
export CO_API_KEY="..."
```

For local development, copy `.env.example` to `.env` and fill in your keys. Never commit `.env`.

### Available Models

#### OpenAI
- `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
- `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`

#### Anthropic
- `claude-sonnet-4-5-20250929`, `claude-opus-4-1-20250805`
- `claude-opus-4-20250514`, `claude-sonnet-4-20250514`
- `claude-3-7-sonnet-latest`, `claude-3-5-haiku-latest`

#### Gemini
- `gemini-2.5-flash`, `gemini-2.5-flash-lite`
- `gemini-2.0-flash`, `gemini-2.0-flash-lite`

#### DeepSeek
- `deepseek-chat`, `deepseek-reasoner`

#### Cohere
- `command-a-03-2025`, `command-r-plus-08-2024`
- `command-r-08-2024`, `command-r7b-12-2024`

## API Reference

### Factory Functions

#### `get_llm_instance(provider: str, model: str) -> LLM`

Create an LLM instance for the specified provider and model.

```python
from majordomo_llm import get_llm_instance

llm = get_llm_instance("openai", "gpt-4o")
```

### LLM Methods

All LLM instances support these async methods:

#### `get_response(user_prompt, system_prompt=None, temperature=0.3, top_p=1.0) -> LLMResponse`

Get a plain text response.

#### `get_json_response(user_prompt, system_prompt=None, temperature=0.3, top_p=1.0) -> LLMJSONResponse`

Get a JSON response (automatically parsed).

#### `get_structured_json_response(response_model, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0) -> LLMStructuredResponse`

Get a response validated against a Pydantic model.

### Response Objects

All response objects include usage metrics:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` / `dict` / `BaseModel` | The response content |
| `input_tokens` | `int` | Number of input tokens |
| `output_tokens` | `int` | Number of output tokens |
| `cached_tokens` | `int` | Number of cached tokens (if applicable) |
| `input_cost` | `float` | Cost for input tokens (USD) |
| `output_cost` | `float` | Cost for output tokens (USD) |
| `total_cost` | `float` | Total cost (USD) |
| `response_time` | `float` | Response time in seconds |

## Advanced Usage

### Automatic Fallback with LLMCascade

Use `LLMCascade` for automatic failover between providers:

```python
from majordomo_llm import LLMCascade

# Providers are tried in order - first is primary, rest are fallbacks
cascade = LLMCascade([
    ("anthropic", "claude-sonnet-4-20250514"),  # Primary
    ("openai", "gpt-4o"),                        # First fallback
    ("gemini", "gemini-2.5-flash"),              # Last resort
])

# If Anthropic fails, automatically tries OpenAI, then Gemini
response = await cascade.get_response("Hello!")
```

All three response methods (`get_response`, `get_json_response`, `get_structured_json_response`) support automatic fallback.

### Direct Provider Access

You can also instantiate providers directly for more control:

```python
from majordomo_llm import Anthropic

llm = Anthropic(
    model="claude-sonnet-4-20250514",
    input_cost=3.0,    # per million tokens
    output_cost=15.0,  # per million tokens
)
```

### Web Search (Anthropic)

Enable web search for supported Claude models:

```python
from majordomo_llm.providers.anthropic import Anthropic

llm = Anthropic(
    model="claude-sonnet-4-5-20250929",
    input_cost=3.0,
    output_cost=15.0,
    use_web_search=True,
)
```

### Request Logging

Log all LLM requests asynchronously to a database with optional storage for request/response bodies. Logging is fire-and-forget and does not block your main request flow.

```python
from majordomo_llm import get_llm_instance
from majordomo_llm.logging import LoggingLLM, PostgresAdapter, S3Adapter

async def main():
    # Create your LLM instance
    llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")

    # Set up database adapter (PostgreSQL, MySQL, or SQLite)
    db = await PostgresAdapter.create(
        host="localhost",
        port=5432,
        database="llm_logs",
        user="postgres",
        password="password",
    )

    # Optional: Set up S3 for storing request/response bodies
    storage = await S3Adapter.create(
        bucket="my-llm-logs",
        prefix="requests",  # optional, defaults to "llm-logs"
    )

    # Wrap your LLM with logging
    logged_llm = LoggingLLM(llm, db, storage)

    # Use as normal - all requests are logged automatically
    response = await logged_llm.get_response("Hello!")

    # Don't forget to close connections when done
    await logged_llm.close()
```

#### Local Development Setup

For local development and testing, use SQLite and local file storage:

```python
from majordomo_llm import get_llm_instance
from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter

async def main():
    llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")

    # SQLite for metrics (auto-creates database and table)
    db = await SqliteAdapter.create("llm_logs.db")

    # Local file storage for request/response bodies
    storage = await FileStorageAdapter.create("./request_logs")

    logged_llm = LoggingLLM(llm, db, storage)
    response = await logged_llm.get_response("Hello!")

    await logged_llm.close()
```

#### API Key Tracking

Track which API key was used for each request with optional human-readable aliases:

```python
from majordomo_llm.providers.anthropic import Anthropic

# Create LLM with API key alias for attribution
llm = Anthropic(
    model="claude-sonnet-4-20250514",
    input_cost=3.0,
    output_cost=15.0,
    api_key_alias="production-team-1",  # Optional human-readable name
)

# The LoggingLLM wrapper automatically logs:
# - api_key_hash: First 16 chars of SHA256 hash (safe for logging)
# - api_key_alias: Your custom name (e.g., "production-team-1")
```

This is useful for:
- Tracking costs per team or application
- Debugging which key was used for specific requests
- Auditing API key usage patterns

#### Database Schema

Create the logging table using the included schema:

```sql
CREATE TABLE IF NOT EXISTS llm_requests (
    request_id VARCHAR(36) PRIMARY KEY,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    response_time FLOAT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    input_cost DECIMAL(10, 8),
    output_cost DECIMAL(10, 8),
    total_cost DECIMAL(10, 8),
    s3_request_key VARCHAR(255),
    s3_response_key VARCHAR(255),
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    api_key_hash VARCHAR(16),
    api_key_alias VARCHAR(100)
);
```

#### Available Adapters

**Database Adapters:**
- **PostgresAdapter** - PostgreSQL via asyncpg
- **MySQLAdapter** - MySQL via aiomysql
- **SqliteAdapter** - SQLite via aiosqlite (great for local development)

**Storage Adapters:**
- **S3Adapter** - AWS S3 via aioboto3
- **FileStorageAdapter** - Local filesystem (great for local development)

## Development

### Setup

```bash
git clone https://github.com/superset-studio/majordomo-llm.git
cd majordomo-llm
uv sync --all-extras
```

### Running Tests

```bash
uv run pytest
```

### Type Checking

```bash
uv run mypy src/majordomo_llm
```

### Linting

```bash
uv run ruff check src/majordomo_llm
```

### Documentation

Build and preview the docs locally:

```bash
uv add --dev mkdocs mkdocs-material mkdocstrings[python] pymdown-extensions
uv run mkdocs serve
```

### Pre-commit Hooks & Checks

Enable local checks (using uvx):

```bash
uvx pre-commit install
uvx pre-commit run --all-files
```

Hooks include private-key detection and basic hygiene checks. See `.pre-commit-config.yaml`.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
