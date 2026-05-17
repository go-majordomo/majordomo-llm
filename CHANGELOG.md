# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-05-16

### Added

- **Amazon Bedrock provider** using the Converse API. Authenticates with long-term Bedrock API keys (`AWS_BEARER_TOKEN_BEDROCK`) and an AWS region (`AWS_REGION` / `AWS_DEFAULT_REGION` or `region=` constructor/factory kwarg). Supports text responses, streaming, raw JSON-schema structured output, and Pydantic-validated structured output via Converse tool calling
- 16 Bedrock models in `llm_config.yaml` (us-east-1 on-demand pricing): Claude 4.x family, Moonshot Kimi K2/K2.5, NVIDIA Nemotron Nano/Nano-3/Super-3, Meta Llama 4 Maverick/Scout, DeepSeek R1/v3.2
- `aioboto3` promoted to a core dependency
- **Deprecated model auto-replacement**: Passing a deprecated model to `get_llm_instance()` automatically resolves to the provider-recommended replacement with a logged warning
- `deprecated_models` section in `llm_config.yaml` mapping old model IDs to replacements for OpenAI, Anthropic, and Gemini
- `LLMResponse.deprecation_warning` field — set when a deprecated model was auto-replaced
- `LLM.requested_model` and `LLM.deprecation_warning` attributes for programmatic detection
- New OpenAI models: `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.4-pro`
- New Anthropic models: `claude-opus-4-6`, `claude-sonnet-4-6`
- New Gemini models: `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`
- `get_json_schema_response()` for raw JSON Schema structured outputs across providers and `LLMCascade`
- Canonical JSON serialization for raw schema responses so equivalent outputs are byte-comparable
- `StructuredOutputUnsupported` error for provider/model structured-output capability failures
- New Anthropic model: `claude-opus-4-7`

### Removed

- Deprecated OpenAI models removed from active config: `gpt-4o`, `gpt-4o-mini`, `gpt-5-pro`, `o1`, `o3-mini`
- Deprecated Anthropic models removed from active config: `claude-3-7-sonnet-20250219`, `claude-3-5-haiku-20241022`, `claude-3-haiku-20240307`
- Deprecated Gemini models removed from active config: `gemini-3-pro-preview`, `gemini-2.0-flash`, `gemini-2.0-flash-lite`

### Changed

- Updated aliases: `fast` → `claude-haiku-4-5-20251001`, `thinking` → `claude-sonnet-4-6`, `smart` → `claude-opus-4-6`, `resilient-sonnet` cascade uses `gpt-4.1` instead of `gpt-4o`
- `Gemini.__init__()` now accepts `supports_temperature_top_p` for constructor consistency across providers
- `get_llm_instance()` accepts a new `region` kwarg, forwarded to the Bedrock provider only
- `examples/shared.py` `PROVIDERS` list now requires a tuple of env vars per entry and includes Bedrock entries (one per upstream model family)

### Fixed

- DeepSeek v4 models (`deepseek-v4-flash`, `deepseek-v4-pro`) no longer send `thinking: disabled` alongside `reasoning_effort: medium`, which the DeepSeek API rejects as mutually exclusive

## [0.3.1] - 2026-02-19

### Added

- `api_key` parameter to `get_llm_instance()` and `LLMCascade` for passing API keys directly instead of relying on environment variables

## [0.2.0] - 2026-02-08

### Added

- Streaming responses via `get_response_stream()` for all providers (OpenAI, Anthropic, Gemini, DeepSeek, Cohere)
- `LLMStreamResponse` async-iterable wrapper with real-time chunk yielding, `.usage` property, and `.collect()` method
- Streaming support in `LLMCascade` with failover on stream creation errors
- Streaming support in `LoggingLLM` with fire-and-forget logging via callbacks
- Streaming demo (`examples/streaming_demo.py`) with real-time output and collect examples

### Fixed

- `claude-haiku-4-5-20251001` config missing `supports_temperature_top_p: false`, causing API errors when both temperature and top_p were sent

## [0.1.6] - 2025-01-31

### Added

- New OpenAI models: `gpt-4o-mini`, `gpt-5-pro`, `o1`, `o3`, `o3-mini`, `o4-mini`
- New Anthropic models: `claude-opus-4-5-20251101`, `claude-haiku-4-5-20251001`, `claude-3-haiku-20240307`
- New Gemini models: `gemini-2.5-pro`, `gemini-3-pro-preview`, `gemini-3-flash-preview`
- Documentation: Basic Usage recipe
- Documentation: Core Concepts section with Structured Outputs, Cost Tracking, and Cascade Failover guides
- Documentation: Expanded homepage with feature overview and quick example
- Documentation: Deprecation automation roadmap (`docs/roadmap/deprecation-automation.md`)

### Changed

- Fixed Anthropic model IDs to use dated snapshots (`claude-3-7-sonnet-20250219`, `claude-3-5-haiku-20241022`) instead of `-latest` aliases
- Organized `llm_config.yaml` with section comments for model families
- Added deprecation comments for Gemini 2.0 models (shutdown March 31, 2026)
- Updated Structured Outputs recipe with comprehensive examples (enums, nested models, constraints)

## [0.1.5] - 2025-01-26

### Added

- Structured response demo (`examples/structured_response_demo.py`) showcasing Pydantic models with enums, nested models, constrained fields, and complex lists
- `inline_schema_refs()` helper to flatten nested JSON schemas by inlining `$defs/$ref` references
- `resolve_api_key()` helper for DRY API key resolution across providers
- `build_schema_prompt()` helper for consistent schema prompt injection
- Shared utilities module (`examples/shared.py`) for common demo functionality

### Changed

- Improved Cohere structured output handling for nested models by flattening schemas
- Refactored provider implementations to use shared helper functions (DRY)
- Moved duplicate `get_json_response()` markdown stripping logic to base class

## [0.1.4] - 2025-01-26

### Added

- API key tracking: `api_key_hash` (SHA256 truncated to 16 chars) and optional `api_key_alias` fields in log entries
- `api_key_alias` parameter to all provider constructors for human-readable key identification
- SQLite adapter (`SqliteAdapter`) for lightweight local development logging
- File storage adapter (`FileStorageAdapter`) for local request/response body storage
- Demo application in `examples/` showcasing multi-provider usage with logging

### Changed

- Updated all database adapter schemas to include `api_key_hash` and `api_key_alias` columns

## [0.1.3] - 2025-01-25

### Added

- Async request logging with `LoggingLLM` wrapper
- PostgreSQL adapter (`PostgresAdapter`) for metrics storage
- MySQL adapter (`MySQLAdapter`) for metrics storage
- S3 adapter (`S3Adapter`) for request/response body storage
- Optional `logging` dependency group: `pip install majordomo-llm[logging]`

## [0.1.2] - 2025-01-25

### Added

- `LLMCascade` class for automatic failover between providers

## [0.1.1] - 2025-01-25

### Added

- DeepSeek provider support (deepseek-chat, deepseek-reasoner models)
- Cohere provider support (Command A, Command R+, Command R, Command R7B models)

### Changed

- Moved LLM configuration from Python dict to external YAML file (llm_config.yaml)
- Added pyyaml as a dependency

## [0.1.0] - 2025-01-25

### Added

- Initial release of majordomo-llm
- Unified interface for multiple LLM providers:
  - OpenAI (GPT-5, GPT-4.1, GPT-4o series)
  - Anthropic (Claude Opus 4, Sonnet 4, Haiku 3.5)
  - Google Gemini (2.0 and 2.5 Flash series)
- Automatic cost tracking for all requests (input/output tokens, USD costs)
- Three response modes:
  - `get_response()` - Plain text responses
  - `get_json_response()` - Parsed JSON responses
  - `get_structured_json_response()` - Pydantic model-validated responses
- Built-in retry logic with exponential backoff (via tenacity)
- Full async/await support for high-performance applications
- Type annotations and py.typed marker for IDE support
- Web search capability for Anthropic Claude models
- Custom exception hierarchy:
  - `MajordomoError` - Base exception
  - `ConfigurationError` - Invalid configuration
  - `ProviderError` - Provider API errors
  - `ResponseParsingError` - Response parsing failures

[Unreleased]: https://github.com/superset-studio/majordomo-llm/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/superset-studio/majordomo-llm/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/superset-studio/majordomo-llm/compare/v0.2.0...v0.3.1
[0.2.0]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/superset-studio/majordomo-llm/releases/tag/v0.1.0
