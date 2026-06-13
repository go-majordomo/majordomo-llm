# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.1] - 2026-06-12

### Changed

- **Gemini catalog refreshed** against https://ai.google.dev/gemini-api/docs/models — added new flagship `gemini-3.5-flash` ($1.50/$9.00, Stable), promoted `gemini-3.1-flash-lite-preview` to the stable name `gemini-3.1-flash-lite` (same pricing), and registered the deprecation mapping so existing callers auto-upgrade. `gemini-3.1-pro-preview` and `gemini-3-flash-preview` remain listed as preview. `examples/shared.py` and `examples/flagship_demo.py` Gemini entries point at `gemini-3.5-flash`; smoke-test `EXTRA_MODELS` swept to the new stable models

## [0.11.0] - 2026-06-01

### Added

- **Bedrock Mantle provider** (`BedrockMantle`) — Anthropic Claude served via AWS-native Anthropic Messages API at `https://bedrock-mantle.{region}.api.aws/anthropic`. Implemented as a thin subclass of `Anthropic`, so Claude's full feature set (structured outputs, prompt caching, extended thinking, tool use, streaming) works out of the box without Converse-shape gymnastics. Authenticates via `AWS_BEARER_TOKEN_BEDROCK` (same bearer token used for the legacy Bedrock Converse path). Region from `AWS_REGION` / `AWS_DEFAULT_REGION` / `region=` constructor arg
- 3 BedrockMantle SKUs in `llm_config.yaml`: Claude Opus 4.8, Opus 4.7, Haiku 4.5 (model IDs use the bare `anthropic.claude-<name>` format). Sonnet 4.6 is not yet hosted on Mantle (returns 404 not_found_error); will be added when AWS lists it

### Changed

- **Bedrock provider scope narrowed to non-Anthropic models.** Anthropic Claude entries removed from the `bedrock:` YAML block — those now live under `bedrock_mantle:`. The remaining Bedrock catalog covers Moonshot Kimi, NVIDIA Nemotron, Meta Llama 4, and DeepSeek-on-Bedrock
- **Removed the Bedrock native Structured Outputs path** (`outputConfig.textFormat.json_schema` via Converse). The supporting allowlist (`_BEDROCK_STRUCTURED_OUTPUTS_SUPPORTED`) and helper (`_bedrock_output_config`) are gone. Rationale: the only beneficiary was Anthropic Claude on Bedrock, which has moved to BedrockMantle where Claude's structured outputs are first-class. Non-Anthropic Bedrock models (Llama 4, Kimi, Nemotron, DeepSeek-on-Bedrock) keep the Converse tool-calling path, which is now the sole Bedrock structured-output mechanism. Eliminates the per-version Anthropic substring maintenance burden — newer Claude releases (Opus 4.8+) just work via BedrockMantle without any allowlist update

### Removed

- `enforce_strict_object_schema` and `strip_unsupported_schema_constraints` are no longer used by the Bedrock provider (they remain in `base.py` and continue to be used by OpenAI strict mode and Cohere respectively)
- `us.anthropic.claude-*` entries from the `bedrock:` YAML block. Migration: use `bedrock_mantle` with `anthropic.claude-*` model IDs (no `us.` prefix, no `-v1` suffix). No backward-compatible alias provided — no users on the previous Bedrock Claude path

### Known limitations

- **Bedrock Nemotron Nano structured output** is grammar-enforced via Bedrock Structured Outputs, but the model can produce malformed JSON on deeply nested or complex schemas (~3+ levels). Simpler schemas pass reliably. For high-reliability structured calls, cascade to a larger model (e.g. `nemotron → claude-haiku`)
- **Together / `json_schema` response format** is supported on a subset of hosted models. The Together provider sends the `json_schema` shape uniformly; models that reject it surface as `ProviderError`. Use the cross-vendor `deepseek-v4-pro` alias to fail over to Fireworks for structured calls on Together-only DeepSeek models

## [0.10.0] - 2026-05-31

### Fixed

- **OpenAI / strict schemas with enum fields**: `inline_schema_refs()` now correctly inlines `$ref` references that have sibling keys (e.g. Pydantic-generated `{"$ref": "...", "description": "..."}` for fields typed with an `Enum`). Previously the `len(obj) == 1` guard left the dangling reference in place after `$defs` was popped, producing `Invalid schema for response_format: reference to component '#/$defs/...' which was not found in the schema` from OpenAI. Field-level descriptions on enum fields are preserved in the inlined output
- **Cohere / strict schema validator** rejects standard JSON Schema constraints (`minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`, `minItems`, `maxItems`, `uniqueItems`, `minLength`, `maxLength`, `pattern`, `format`). The Cohere provider now strips these recursively before sending so Pydantic models using `Field(ge=, le=, min_length=, ...)` work as-is. The stripping logic was promoted to a shared `strip_unsupported_schema_constraints` helper in `base.py` and is reused by the Bedrock Structured Outputs path
- **Bedrock / Llama 4 structured output**: `us.meta.llama4-*` models reject `toolConfig.toolChoice.tool` in the Converse API. The Bedrock provider now omits the `toolChoice` field for Llama 4 model IDs while still exposing the tool, relying on the system-prompt instruction to steer the model toward the tool call
- **Bedrock / Nemotron structured output (and grammar-enforced JSON for all supported models)**: previously, Bedrock structured output went exclusively through Converse tool calling, which produced opaque `InternalServerException` errors on NVIDIA Nemotron Nano. The Bedrock provider now uses native Bedrock Structured Outputs (`outputConfig.textFormat.json_schema`) for the supported model families — Anthropic Claude, NVIDIA Nemotron Nano, Qwen3, Google Gemma, Mistral — where Bedrock compiles the schema into a grammar and enforces it during generation. Tool calling remains the fallback path for models outside that allowlist (Llama 4, Moonshot Kimi K2.5, DeepSeek v3.2). Schemas are auto-normalized with `additionalProperties: false` and full `required` lists (the new shared `enforce_strict_object_schema` helper, previously `_enforce_openai_strict_schema`), and the same grammar-incompatible constraints stripped by Cohere (`minimum`, `maximum`, `minItems`, etc.) are stripped before being sent to Bedrock
- **DeepSeek / structured output uses correct response_format**: DeepSeek's API supports only `response_format={"type": "json_object"}` (per https://api-docs.deepseek.com/guides/json_mode); the previous `json_schema` request shape was rejected by every DeepSeek SKU with `"This response_format type is unavailable now"`. The DeepSeek provider now uses `json_object` mode and injects the schema into the system prompt via `build_schema_prompt()`, restoring structured output across `deepseek-chat`, `deepseek-v4-pro`, and `deepseek-v4-flash`
- **Fireworks / `reasoning_effort` + `thinking` conflict**: Fireworks rejects requests that specify both `reasoning_effort` and `thinking` (`cannot specify both 'thinking' and 'reasoning_effort'`), which broke the `deepseek-v4-pro-reasoning` and `deepseek-v4-pro-hard` profile aliases (both set both fields). The Fireworks provider now collapses the two fields: `thinking="disabled"` takes precedence (explicit opt-out wins); otherwise `reasoning_effort` is sent alone since it already implies thinking is on. Together still accepts both fields and is unaffected

### Added

- **Fireworks AI provider** (`Fireworks`) via the OpenAI-compatible `https://api.fireworks.ai/inference/v1` endpoint. Supports text, streaming, raw JSON-schema structured output, and Pydantic-validated structured output. Authenticates with `FIREWORKS_API_KEY`
- **Together AI provider** (`Together`) via the OpenAI-compatible `https://api.together.xyz/v1` endpoint. Same capability surface as Fireworks. Authenticates with `TOGETHER_API_KEY`. Note: Together's `json_schema` response format is supported on a subset of hosted models; the request uses the standard shape uniformly and surfaces model-side rejections as `ProviderError`
- 4 Fireworks serverless SKUs in `llm_config.yaml`: DeepSeek-V4-Pro, Kimi-K2.5, Kimi-K2.6, GLM-5.1
- 7 Together serverless SKUs in `llm_config.yaml`: DeepSeek-V4-Pro, Kimi-K2.6, Qwen3.6-Plus, Qwen3.5-9B, Qwen3-235B-A22B-fp8-tput, GLM-5.1, GLM-5
- `reasoning_effort` and `thinking` constructor kwargs on `Fireworks` and `Together`, mirroring the `DeepSeek` provider. Validated against the same effort/thinking value sets; forwarded via top-level `reasoning_effort` and `extra_body={"thinking": {"type": ...}}` respectively. Plumbed through `get_llm_instance()` from YAML attributes
- **Multi-profile model registration**: a `llm_config.yaml` model entry may declare a `model:` field that overrides the API model ID, decoupling it from the YAML key. Lets the same upstream SKU be registered under multiple profile names with different reasoning configs
- Three DeepSeek-V4-Pro reasoning profiles (`-reasoning`, `-hard`) registered under `deepseek`, `fireworks`, and `together` using the new `model:` override
- Cross-vendor cascade aliases — `deepseek-v4-pro`, `kimi-k2.6`, `glm-5.1` (Fireworks → Together), and `deepseek-v4-pro-reasoning` / `deepseek-v4-pro-hard` (Fireworks → Together → Anthropic Sonnet/Opus as quality safety net)
- `flagship_demo.py` expanded to compare closed-source frontiers (Opus 4.7, GPT-5.5, Gemini 3.1 Pro Preview, DeepSeek-Reasoner) side-by-side with DeepSeek-V4-Pro at three reasoning profiles across both Fireworks and Together

## [0.9.1] - 2026-05-18

### Fixed

- **OpenAI structured outputs** now normalize Pydantic-derived schemas to satisfy OpenAI's strict-mode requirements. Every object node in the schema gets `additionalProperties: false` and every property is added to `required`, with `$ref`/`$defs` inlined first. Previously, calling `get_structured_json_response()` against any current OpenAI model failed with `Invalid schema for response_format ...: 'additionalProperties' is required to be supplied and to be false`

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
