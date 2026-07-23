# majordomo-llm Examples

This directory contains example applications demonstrating majordomo-llm features.

## Structured Response Demo

The `structured_response_demo.py` script showcases the `get_structured_json_response()` method with various Pydantic models:

- **Sentiment Analysis** - Simple model with Enum field
- **Text Analysis** - Nested models with entity extraction
- **Code Review** - Constrained integer fields and booleans
- **Product Recommendations** - Complex nested lists with validation

### Run

```bash
uv run python examples/structured_response_demo.py
```

### Example Output

```
Demo 1: Sentiment Analysis (with Enum)
Result (SentimentAnalysis):
  Sentiment: positive
  Confidence: 95.00%
  Reasoning: The text expresses enthusiasm with words like "thrilled" and "exceeded expectations"
```

## Streaming Demo

The `streaming_demo.py` script showcases the `get_response_stream()` method:

- **Real-time streaming** - Chunks printed as they arrive with time-to-first-chunk metrics
- **Collect into LLMResponse** - Using `.collect()` to consume the stream and get the full response

### Run

```bash
uv run python examples/streaming_demo.py
```

### Example Output

```
Demo 1: Streaming with real-time output

  [anthropic/claude-3-5-haiku-latest]
  The sky appears blue because of Rayleigh scattering...
  Time to first chunk: 0.34s | Total: 1.12s
  Tokens: 28 in / 45 out | Cost: $0.000159

Demo 2: Collect stream into LLMResponse

  [anthropic/claude-3-5-haiku-latest]
  Content: The three primary colors are red, blue, and yellow.
  Tokens: 22 in / 15 out | Cost: $0.000041
```

## Prompt Caching Demo

The `prompt_caching_demo.py` script showcases both prompt caching flavors side
by side, using a large reused system prompt as the cacheable prefix:

- **Explicit caching** (Anthropic `claude-sonnet-5` / `claude-opus-4-8-fast`,
  Bedrock Mantle) — this library controls the `cache_control` breakpoint, so it
  demonstrates cache **creation** (`cache_creation_tokens` > 0 on the cold
  call), cache **read** (`cached_tokens` > 0 on the warm call), and the
  `use_prompt_caching=False` toggle on `get_llm_instance` that suppresses the
  breakpoint entirely.
- **Automatic caching** (OpenAI `gpt-5.6-luna`, Gemini `gemini-3.6-flash`,
  DeepSeek `deepseek-v4-flash`) — the provider caches repeated prefixes
  server-side; there is no creation step or toggle, but `cached_tokens` populate
  on the warm call and bill at the discounted `cached_input_cost` rate.

Cache-aware cost accounting is shown for both: `input_cost` folds in cache
read/write tokens (additive for the explicit providers, subset re-pricing for
the automatic ones) using the rates in `llm_config.yaml`.

### Run

```bash
uv run python examples/prompt_caching_demo.py
uv run python examples/prompt_caching_demo.py --provider anthropic
```

### Example Output

```
  [anthropic/claude-sonnet-5]  (explicit cache-control), system prompt ~24200 chars

    Flow A — reuse the same system prompt across two calls:
    Call 1 (cold — expect cache WRITE > 0):
      Tokens: 24 in / 18 out | cache write 3050 / cache read 0
      Cost: $0.011... (input $0.011... + output $0.000...)
    Call 2 (warm — expect cache READ > 0):
      Tokens: 26 in / 41 out | cache write 0 / cache read 3050
      Cost: $0.001... (input $0.000... + output $0.000...)
      Cache hit: 3050 tokens read; prompt-side savings vs. uncached ~= $0.008235

    Flow B — caching OFF (use_prompt_caching=False):
    Call (expect cache write/read == 0):
      Tokens: 3074 in / 18 out | cache write 0 / cache read 0
      Cost: $0.009... (input $0.009... + output $0.000...)

  [openai/gpt-5.6-luna]  (automatic caching), system prompt ~24200 chars

    Flow A — reuse the same system prompt across two calls:
    Call 1 (cold — expect cache read 0):
      Tokens: 3072 in / 20 out | cache write 0 / cache read 0
    Call 2 (warm — expect cache READ > 0):
      Tokens: 3074 in / 44 out | cache write 0 / cache read 2944
      Cache hit: 2944 tokens read; prompt-side savings vs. uncached ~= $0.002650

    (No use_prompt_caching toggle — this provider caches automatically ...)
```

## Demo: Multi-Provider Comparison with Logging

The `demo.py` script showcases:

- Running the same prompts across multiple LLM providers (OpenAI, Anthropic, Gemini, DeepSeek, Cohere)
- Automatic request logging to SQLite with API key hash tracking
- Local file storage for request/response bodies
- Cost and performance comparison across providers

### Setup

1. Install dependencies with the logging extras:

```bash
uv sync --all-extras
```

2. Set API keys for the providers you want to test. Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=sk-...
CO_API_KEY=...
```

Or export them in your shell. You don't need all keys - the demo will skip providers without keys.

### Run

```bash
uv run python examples/demo.py
```

### Output

The demo will:

1. Run 3 prompts (code, content, customer support) against each available provider
2. Display responses, token counts, costs, and timing for each
3. Print a summary table from the logged metrics
4. Save all request/response bodies as JSON files in `examples/request_logs/`

### Files Created

After running, you'll have:

- `llm_logs.db` - SQLite database with request metrics
- `request_logs/` - Directory with JSON files for each request/response

You can query the SQLite database directly:

```bash
# Basic query
sqlite3 examples/llm_logs.db "SELECT provider, model, total_cost, response_time FROM llm_requests"

# Query with API key tracking (api_key_hash is first 16 chars of SHA256)
sqlite3 examples/llm_logs.db "SELECT provider, model, api_key_hash, api_key_alias FROM llm_requests"
```

## Prompts

The `prompts.json` file contains sample prompts across three domains:

- **code-generation**: Rust ownership explanation
- **content-generation**: Marketing tagline creation
- **customer-support**: Ticket classification

Feel free to modify or add prompts to test different scenarios.
