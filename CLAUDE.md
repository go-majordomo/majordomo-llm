# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install dependencies (includes dev + logging extras)
uv sync --all-extras

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_anthropic.py

# Run a single test
uv run pytest tests/test_anthropic.py::test_anthropic_response

# Type checking
uv run mypy src/majordomo_llm

# Linting
uv run ruff check src/majordomo_llm

# Fix linting issues
uv run ruff check --fix src/majordomo_llm
```

## Architecture

Unified async interface for LLM providers (OpenAI, Anthropic, Gemini, DeepSeek, Cohere) with cost tracking and structured output support.

### Core Components

- **`base.py`**: Abstract `LLM` base class defining the interface (`get_response`, `get_json_response`, `get_structured_json_response`). Response dataclasses (`LLMResponse`, `LLMJSONResponse`, `LLMStructuredResponse`) inherit from `Usage` for cost/token tracking.

- **`factory.py`**: `get_llm_instance(provider, model)` factory function. Loads model configs from `llm_config.yaml` (costs per million tokens, feature flags like `supports_temperature_top_p`).

- **`llm_config.yaml`**: Model configuration (costs, feature flags). Add new models here; the factory will pick them up automatically.

- **`cascade.py`**: `LLMCascade` wraps multiple providers for automatic failover. Tries providers in order; catches `ProviderError` and falls back to next.

- **`providers/`**: Provider implementations extending `LLM`. Each provider:
  - Uses its native async client (e.g., `anthropic.AsyncAnthropic`)
  - Implements `get_response()` with automatic retries via tenacity
  - Overrides `_get_structured_response()` for provider-specific structured output (Anthropic uses tool calling, others use response schemas)

- **`logging/`**: Optional request logging subsystem (requires `[logging]` extra):
  - `LoggingLLM`: Wrapper that logs requests fire-and-forget (non-blocking)
  - `interfaces.py`: `DatabaseAdapter` and `StorageAdapter` ABCs
  - `adapters/`: Database (`PostgresAdapter`, `MySQLAdapter`, `SqliteAdapter`) and storage (`S3Adapter`, `FileStorageAdapter`) implementations

- **`examples/`**: Demo app showing multi-provider usage with logging. Run with `uv run python examples/demo.py`

- **`exceptions.py`**: Exception hierarchy: `MajordomoError` (base) → `ConfigurationError`, `ProviderError`, `ResponseParsingError`

### Key Patterns

- All LLM methods are async and return response objects with embedded usage metrics
- Retry logic: `@retry(wait=wait_random_exponential(min=0.2, max=1), stop=stop_after_attempt(3))`
- Costs calculated per million tokens using `TOKENS_PER_MILLION = 1_000_000`
- Pydantic models for structured output schemas via `model_json_schema()`
- Provider errors are wrapped in `ProviderError` with `original_error` attribute
- API key hashing: `_hash_api_key()` uses SHA256 truncated to 16 hex chars for safe logging
- All providers accept `api_key_alias` for human-readable key identification in logs

### Environment Variables

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GEMINI_API_KEY` - Google Gemini API key
- `DEEPSEEK_API_KEY` - DeepSeek API key
- `CO_API_KEY` - Cohere API key
