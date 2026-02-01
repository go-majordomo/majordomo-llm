# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.6...HEAD
[0.1.6]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/superset-studio/majordomo-llm/releases/tag/v0.1.0
