"""Tests for the hook primitive (HookPipeline, RegexHook, LLMJudgeHook)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from majordomo_llm import (
    HookBlocked,
    HookContext,
    HookOutcome,
    HookPipeline,
    HookVerdict,
    LLMJudgeHook,
    RegexHook,
)


@pytest.fixture
def echo_call():
    async def _call(prompt: str) -> str:
        return f"echo:{prompt}"

    return _call


class _StubHook:
    """Hook whose phase methods return a caller-supplied outcome.

    Used to drive specific pipeline branches without standing up a full
    regex/judge configuration each time.
    """

    def __init__(self, name: str, *, before=None, after=None):
        self.name = name
        self._before = before
        self._after = after
        self.before_called_with: list[tuple[str, HookContext]] = []
        self.after_called_with: list[tuple[str, str, HookContext]] = []

    async def before_call(self, prompt: str, ctx: HookContext) -> HookOutcome:
        self.before_called_with.append((prompt, ctx))
        if self._before is None:
            return HookOutcome.pass_through(self.name)
        return self._before(prompt, ctx)

    async def after_call(self, prompt: str, response: str, ctx: HookContext) -> HookOutcome:
        self.after_called_with.append((prompt, response, ctx))
        if self._after is None:
            return HookOutcome.pass_through(self.name)
        return self._after(prompt, response, ctx)


class TestHookPipelineBefore:
    async def test_blocks_before_call_is_invoked(self, echo_call):
        """A block in before_call raises HookBlocked before the LLM call runs."""
        hook = _StubHook("blocker", before=lambda p, _: HookOutcome.block("blocker", "no"))
        call = AsyncMock(side_effect=echo_call)
        pipeline = HookPipeline([hook])

        with pytest.raises(HookBlocked) as exc_info:
            await pipeline.run("prompt", call)

        assert call.await_count == 0
        assert exc_info.value.verdicts[0].hook_name == "blocker"
        assert exc_info.value.verdicts[0].action_taken == "block"

    async def test_before_call_redacts_modifies_prompt_to_call(self):
        """Redacted prompt flows into the LLM call argument."""
        hook = _StubHook(
            "redactor",
            before=lambda p, _: HookOutcome.redact("redactor", "REDACTED", "match"),
        )
        seen: list[str] = []

        async def call(prompt: str) -> str:
            seen.append(prompt)
            return "ok"

        pipeline = HookPipeline([hook])
        await pipeline.run("original", call)
        assert seen == ["REDACTED"]


class TestHookPipelineAfter:
    async def test_after_call_redacts_modifies_returned_response(self, echo_call):
        hook = _StubHook(
            "redactor",
            after=lambda p, r, _: HookOutcome.redact("redactor", "CLEAN", "match"),
        )
        pipeline = HookPipeline([hook])
        result = await pipeline.run("prompt", echo_call)
        assert result == "CLEAN"

    async def test_after_call_block_raises_after_call(self, echo_call):
        hook = _StubHook("blocker", after=lambda p, r, _: HookOutcome.block("blocker", "no"))
        pipeline = HookPipeline([hook])
        with pytest.raises(HookBlocked):
            await pipeline.run("prompt", echo_call)


class TestHookPipelineOrdering:
    async def test_modifications_propagate_through_hooks_in_order(self):
        """Later hooks see modifications from earlier hooks within a phase."""
        first = _StubHook(
            "first",
            before=lambda p, _: HookOutcome.redact("first", p + "A", "step"),
        )
        second = _StubHook(
            "second",
            before=lambda p, _: HookOutcome.redact("second", p + "B", "step"),
        )
        seen: list[str] = []

        async def call(prompt: str) -> str:
            seen.append(prompt)
            return "done"

        pipeline = HookPipeline([first, second])
        await pipeline.run("p", call)
        assert seen == ["pAB"]


class TestHookPipelineExceptions:
    async def test_hook_raising_non_blocked_is_swallowed_as_pass(self, echo_call):
        """A buggy hook must not break the underlying LLM call."""

        def explode(prompt, ctx):
            raise RuntimeError("boom")

        hook = _StubHook("boom", before=explode)
        pipeline = HookPipeline([hook])
        result = await pipeline.run("prompt", echo_call)
        assert result == "echo:prompt"

    async def test_on_verdicts_exceptions_are_swallowed(self, echo_call):
        def explode(_request_id, _verdicts):
            raise RuntimeError("boom")

        pipeline = HookPipeline([_StubHook("noop")], on_verdicts=explode)
        result = await pipeline.run("prompt", echo_call)
        assert result == "echo:prompt"


class TestHookPipelineVerdictRecording:
    async def test_on_verdicts_fires_on_success_path(self, echo_call):
        recorded: list[tuple] = []

        def record(request_id, verdicts):
            recorded.append((request_id, list(verdicts)))

        pipeline = HookPipeline([_StubHook("noop")], on_verdicts=record)
        await pipeline.run("prompt", echo_call)
        assert len(recorded) == 1
        request_id, verdicts = recorded[0]
        assert all(isinstance(v, HookVerdict) for v in verdicts)
        # before-pass + after-pass for one hook
        assert len(verdicts) == 2

    async def test_on_verdicts_fires_on_block_path(self):
        """Verdicts up to and including the blocker are emitted."""
        recorded: list[list[HookVerdict]] = []

        def record(_request_id, verdicts):
            recorded.append(list(verdicts))

        hook = _StubHook("b", before=lambda p, _: HookOutcome.block("b", "stop"))
        pipeline = HookPipeline([hook], on_verdicts=record)
        with pytest.raises(HookBlocked):
            await pipeline.run("prompt", AsyncMock())

        assert len(recorded) == 1
        assert recorded[0][-1].action_taken == "block"

    async def test_on_verdicts_supports_async_callback(self, echo_call):
        recorded: list[list[HookVerdict]] = []

        async def record(_request_id, verdicts):
            recorded.append(list(verdicts))

        pipeline = HookPipeline([_StubHook("noop")], on_verdicts=record)
        await pipeline.run("prompt", echo_call)
        assert recorded


class TestHookContext:
    async def test_caller_metadata_reaches_both_phases(self, echo_call):
        seen: list[HookContext] = []

        def grab_before(prompt, ctx):
            seen.append(ctx)
            return HookOutcome.pass_through("h")

        def grab_after(prompt, response, ctx):
            seen.append(ctx)
            return HookOutcome.pass_through("h")

        hook = _StubHook("h", before=grab_before, after=grab_after)
        pipeline = HookPipeline([hook])
        await pipeline.run("prompt", echo_call, caller_metadata={"feature": "x"})

        assert len(seen) == 2
        assert seen[0].phase == "before"
        assert seen[1].phase == "after"
        assert seen[0].caller_metadata == {"feature": "x"}
        assert seen[1].caller_metadata == {"feature": "x"}
        assert seen[0].request_id == seen[1].request_id


class TestRegexHook:
    async def test_block_action_raises(self, echo_call):
        hook = RegexHook(name="ssn", pattern=r"\d{3}-\d{2}-\d{4}", action="block")
        pipeline = HookPipeline([hook])
        with pytest.raises(HookBlocked):
            await pipeline.run("user 123-45-6789", echo_call)

    async def test_warn_action_passes_text_through(self):
        hook = RegexHook(name="ssn", pattern=r"\d{3}-\d{2}-\d{4}", action="warn")
        pipeline = HookPipeline([hook])

        async def call(prompt: str) -> str:
            return prompt

        result = await pipeline.run("user 123-45-6789", call)
        assert result == "user 123-45-6789"

    async def test_redact_action_replaces_match(self):
        hook = RegexHook(
            name="ssn",
            pattern=r"\d{3}-\d{2}-\d{4}",
            action="redact",
            redaction="[SSN]",
            phase="before",
        )
        seen: list[str] = []

        async def call(prompt: str) -> str:
            seen.append(prompt)
            return "ok"

        pipeline = HookPipeline([hook])
        await pipeline.run("user 123-45-6789", call)
        assert seen == ["user [SSN]"]

    async def test_phase_after_skips_before(self):
        hook = RegexHook(
            name="boom",
            pattern=r"x",
            action="block",
            phase="after",
        )
        called: list[str] = []

        async def call(prompt: str) -> str:
            called.append(prompt)
            return "clean"

        pipeline = HookPipeline([hook])
        result = await pipeline.run("x is here", call)
        # before-phase skipped; response has no match → success
        assert result == "clean"
        assert called == ["x is here"]


class TestLLMJudgeHook:
    async def test_pass_verdict_passes_through(self, echo_call):
        async def judge(_rendered: str) -> str:
            return json.dumps({"verdict": "pass", "reason": ""})

        hook = LLMJudgeHook(
            name="judge",
            judge_call=judge,
            judge_prompt="evaluate {response}",
            phase="after",
        )
        pipeline = HookPipeline([hook])
        result = await pipeline.run("prompt", echo_call)
        assert result == "echo:prompt"

    async def test_fail_verdict_with_block_raises(self, echo_call):
        async def judge(_rendered: str) -> str:
            return json.dumps({"verdict": "fail", "reason": "off-topic"})

        hook = LLMJudgeHook(
            name="judge",
            judge_call=judge,
            judge_prompt="evaluate {response}",
            phase="after",
            action="block",
        )
        pipeline = HookPipeline([hook])
        with pytest.raises(HookBlocked):
            await pipeline.run("prompt", echo_call)

    async def test_timeout_passes_through(self, echo_call):
        async def slow_judge(_rendered: str) -> str:
            await asyncio.sleep(1.0)
            return json.dumps({"verdict": "fail"})

        hook = LLMJudgeHook(
            name="judge",
            judge_call=slow_judge,
            judge_prompt="{response}",
            phase="after",
            timeout_seconds=0.01,
        )
        pipeline = HookPipeline([hook])
        result = await pipeline.run("prompt", echo_call)
        assert result == "echo:prompt"

    async def test_unparseable_judge_output_passes_through(self, echo_call):
        async def bad_judge(_rendered: str) -> str:
            return "not json"

        hook = LLMJudgeHook(
            name="judge",
            judge_call=bad_judge,
            judge_prompt="{response}",
            phase="after",
        )
        pipeline = HookPipeline([hook])
        result = await pipeline.run("prompt", echo_call)
        assert result == "echo:prompt"

    async def test_judge_exception_passes_through(self, echo_call):
        async def exploding_judge(_rendered: str) -> str:
            raise RuntimeError("upstream down")

        hook = LLMJudgeHook(
            name="judge",
            judge_call=exploding_judge,
            judge_prompt="{response}",
            phase="after",
        )
        pipeline = HookPipeline([hook])
        result = await pipeline.run("prompt", echo_call)
        assert result == "echo:prompt"


class TestArbitraryCall:
    async def test_pipeline_works_with_arbitrary_call(self):
        """The pipeline does not depend on LLMCascade."""
        external = MagicMock(return_value="external response")

        async def call(prompt: str) -> str:
            return external(prompt)

        pipeline = HookPipeline([_StubHook("noop")])
        result = await pipeline.run("prompt", call)
        assert result == "external response"
        external.assert_called_once_with("prompt")
