# Reasoning & Thinking

## Why This Needs a Decision

Most current models reason before they answer, and the reasoning is billed as output
tokens. On the open-weight hosts it is on by default and usually at the top tier, so a
call that sends no reasoning option pays for the most the model will do.

A one-word classification makes the cost visible. Each row is the same prompt, sent
once with nothing set and once with `reasoning_effort: low`:

| Host | Model | Output tokens, nothing sent | Output tokens, `low` |
|------|-------|-----------------------------|----------------------|
| Fireworks | `glm-5p3` | 86 | 7 |
| Fireworks | `kimi-k3` | 135 | 47 |
| Together | `zai-org/GLM-5.3` | 114 | 4 |
| Together | `moonshotai/Kimi-K3` | 78 | 15 |
| Baseten | `moonshotai/Kimi-K3` | 92 | 15 |

The answer was "billing" every time. The difference is reasoning the caller never sees.

Those numbers are one sample of one trivial prompt, measured on 2026-10-11. A second
run the same day moved individual figures by up to 2x but kept every gap in this table.
One apparent saving did not survive it: `low` looked like it halved reasoning on
Together's two DeepSeek models in the first run and made no difference in the second,
so those two are not pinned. Treat a single run as a hint, not a result. The numbers
show what reasoning costs in tokens. They say nothing about what lower effort does to answer
quality on a hard task, which is the other half of the trade.

## What majordomo-llm Does by Default

Reasoning settings live per model in `llm_config.yaml`, as `reasoning_effort` and
`thinking`. The library sets `reasoning_effort: low` on every open-weight model where
`low` measurably lowers reasoning, and sends nothing on the rest:

| Host | Sends `reasoning_effort: low` | Sends nothing |
|------|-------------------------------|---------------|
| Fireworks | `glm-5p3`, `kimi-k3`, `qwen3p8-2p4t-a95b`, `qwen3p8-max` | `inkling` |
| Together | `GLM-5.3`, `GLM-5.3-Flash`, `Kimi-K3`, `Qwen3.8-2.4T-A95B` | `DeepSeek-V4-Pro-0813`, `DeepSeek-V4.1-Flash`, `GLM-5.2`, `Qwen3.5-9B`, `Inkling`, and the four streaming-only Qwen models |
| Baseten | `GLM-5.3`, `GLM-5.3-Flash`, `Kimi-K3` | `DeepSeek-V4-Pro-0813`, `DeepSeek-V4.1-Flash`, `GLM-5.2` |

A model in the right-hand column still reasons at the host's default, with one
exception: Baseten's `GLM-5.2`, where reasoning is opt-in and stays off.

Anthropic models carry no default. Reasoning there is configured through named
profiles (`claude-opus-4-8-fast`, `-medium`, `-deep`) or left to the model's own
default. With nothing sent, the model decides for itself whether to think, and bills
any thinking as output. On a one-word classification, `claude-opus-5` and
`claude-fable-5` thought and the 5.5 models did not; on a Markdown rewrite with about
4,300 tokens of text, `claude-sonnet-5-5` used 8,700 output tokens. `claude-opus-5-5`,
`claude-fable-5-1` and `claude-fable-5` cannot have thinking turned off. OpenAI, Gemini, Cohere, Bedrock and Bedrock Mantle have no reasoning setting in
this library.

Having no setting does not mean a model does not reason. Gemini 3 models think on
every request and bill that thinking at the output rate, while reporting it apart
from the visible answer. The library adds the two, so `output_tokens` and
`output_cost` on a Gemini response cover both; on short answers the thinking is
usually the larger part.

## Choosing for Yourself

Pass `reasoning_effort` or `thinking` to `get_llm_instance()` to override the config
for that instance:

```python
from majordomo_llm import get_llm_instance

# More reasoning than the library default
llm = get_llm_instance("together", "moonshotai/Kimi-K3", reasoning_effort="high")

# No reasoning at all
llm = get_llm_instance("baseten", "deepseek-ai/DeepSeek-V4.1-Flash", thinking="disabled")

# Send no reasoning option, and let the host decide
llm = get_llm_instance(
    "fireworks", "accounts/fireworks/models/kimi-k3", reasoning_effort="default"
)
```

Four rules govern the override:

1. **It is one override, not two.** Passing either argument replaces *both* configured
   values. `thinking="disabled"` on a model configured with `reasoning_effort: low`
   sends only `thinking`, never the two together.
2. **`reasoning_effort="default"` sends nothing.** It removes the library's `low` and
   leaves the host's own default in force.
3. **`thinking="disabled"` means off on every model.** It is sent as that model's own
   spelling of "off"; see [Turning Thinking Off](#turning-thinking-off).
4. **Other values are passed through as-is.** The library checks that a value is one a host
   could accept (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` for effort;
   `enabled` or `disabled` for thinking on the open-weight hosts). Whether a particular
   model accepts it is up to the host, and a rejected value comes back as a
   `ProviderError`.

The override is supported on `anthropic`, `fireworks`, `together` and `baseten`. On any
other provider it raises `ConfigurationError` rather than being silently ignored.

## Turning Thinking Off

Models spell "off" differently, and some have no "off". `thinking="disabled"` is the
one spelling callers use; `get_llm_instance()` sends whatever the model needs, read from
the `thinking_off` key of its `llm_config.yaml` entry. A `thinking: disabled` set in
the config itself is translated the same way, and so is a cascade member's
`{"thinking": "disabled"}`.

| `thinking_off` | Sent for `thinking="disabled"` | Models |
|----------------|--------------------------------|--------|
| absent (`disabled`) | `{"type": "disabled"}` | Every model not listed below. Measured to work on `claude-haiku-5-5`, `claude-opus-5`, `claude-opus-4-8`, `claude-sonnet-5`, Kimi K3, Qwen 3.8, Together's `GLM-5.2`, and the DeepSeek V4 models on Together and Baseten; the rest have not been probed |
| `between_tools` | `{"type": "between_tools"}` | `claude-sonnet-5-5`, which rejects `disabled` with a 400 |
| `default` | nothing | Baseten's `GLM-5.2`, which reasons only when asked |
| `unsupported` | raises `ConfigurationError` | `claude-opus-5-5`, `claude-fable-5-1` and `claude-fable-5`, which reject both `disabled` and `between_tools`; `glm-5p3` and `inkling` on Fireworks; `GLM-5.3`, `Qwen3.5-9B` and `Inkling` on Together |

An `unsupported` model raises rather than accepting a value that would not turn
reasoning off, so a caller asking for no thinking finds out when the instance is built,
not from the bill. On those Claude models the API's advice is to keep thinking
`adaptive` and lower `reasoning_effort` instead.

`between_tools` on `claude-sonnet-5-5` is "off" before the answer, not everywhere.
In the API's words, the model "does not think before responding", but the short
updates it writes between tool calls come back as thinking blocks. With web search
on, those updates are still billed as output; the library drops them from `content`.

The Anthropic rows come from `scripts/check_anthropic_thinking.py`, which sends each
registered Claude model `disabled` and `between_tools` and prints which one it accepts
next to the configured value, along with the API's error message for each one it
rejects. Run it when adding a Claude model.

Constructing a provider class directly (`Anthropic(thinking=...)`) bypasses the
translation and sends the value as given.

## Values Are Not Portable

The same value does different things on different hosts, and sometimes on different
models from one host. These are measured, not taken from vendor documentation:

| Behaviour | Where |
|-----------|-------|
| `low` cuts reasoning sharply | GLM 5.3 and Kimi K3 on all three hosts |
| `low` has no effect | Together `GLM-5.2`, `Qwen3.5-9B` and both DeepSeek models; Baseten's two DeepSeek models |
| Too noisy to call | Inkling on Fireworks and Together, where output for the same setting varied by 2–3x between runs |
| Any effort value switches reasoning **on** | Baseten `GLM-5.2` (opt-in): 1 output token by default, 259 with `low` |
| Reasoning cannot be disabled | GLM 5.3 on Fireworks (rejected: "thinking-only model") and on Together (accepted, but no effect) |
| `high` and `none` are rejected with HTTP 400 | Together `Qwen3.8-2.4T-A95B` |
| `thinking="disabled"` turns reasoning off | Kimi K3 everywhere; Qwen 3.8 on Fireworks and Together; Together `GLM-5.2` and both DeepSeek models; Baseten's DeepSeek models |
| Only `reasoning_effort="none"` turns it off | Together `Inkling` |
| Cannot be turned off through this library | Together `Qwen3.5-9B`, which needs `reasoning: {"enabled": false}` |

So there is no single value that means "cheap" everywhere. If you move a workload
between hosts, or build an `LLMCascade` across them, check the row for each member.
The per-host measurements are recorded as comments on the `fireworks`, `together` and
`baseten` blocks of `llm_config.yaml`.

## Seeing the Cost

Reasoning tokens are part of `response.output_tokens` and `response.output_cost`.
The library does not report them separately, so the way to see what reasoning costs
on your own prompts is to run the same call at two settings and compare:

```python
for effort in ("low", "high"):
    llm = get_llm_instance("together", "zai-org/GLM-5.3", reasoning_effort=effort)
    response = await llm.get_response(prompt)
    print(effort, response.output_tokens, f"${response.total_cost:.6f}")
```

With reasoning on, the output cap covers reasoning and answer together. A response cut
off at the cap raises `ResponseTruncatedError`; see `max_tokens` in the README.

## Design Notes

- **Why `low` is the default.** The alternative was to send nothing and inherit each
  host's default, which is the top tier on most of these models. A default that costs
  several times more than necessary for routine calls is the worse surprise, and the
  override makes the other choice one argument away.
- **Why the default is set per model.** A blanket `low` would have switched reasoning
  on for Baseten's `GLM-5.2` and done nothing for six other models. Each entry reflects
  what was measured for that model on that host.
- **Why only "off" is translated.** Mapping every value onto every host would need a
  table that goes stale as fast as the hosts change their models. "Off" is the one
  request whose meaning is the same everywhere, and getting it wrong costs money
  silently, so it alone is translated, from a per-model key set only where the
  behaviour was measured. Every other value is passed through, which keeps the
  library honest about what it knows.
- **These hosts change often.** Re-measure before relying on a row above for anything
  cost-sensitive.

## Re-measuring

`scripts/check_reasoning.py` produced every number on this page. It calls each
registered Fireworks, Together and Baseten model under seven settings and prints output
tokens, reasoning tokens and latency, marking the row that `llm_config.yaml` currently
sends:

```bash
uv run python scripts/check_reasoning.py
uv run python scripts/check_reasoning.py --provider together
uv run python scripts/check_reasoning.py --provider baseten --model zai-org/GLM-5.2
```

It runs inference, seven small calls per model, so a full sweep costs a few cents. Run
it when adding a model to one of these hosts, and before setting or changing a
`reasoning_effort` in the config.
