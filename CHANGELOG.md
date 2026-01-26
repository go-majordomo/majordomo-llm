# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/superset-studio/majordomo-llm/releases/tag/v0.1.0
