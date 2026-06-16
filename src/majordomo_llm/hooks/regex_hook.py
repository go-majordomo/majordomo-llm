"""Regex-based hook implementation."""

import re
from dataclasses import dataclass, field
from typing import Literal

from majordomo_llm.hooks.protocol import HookContext, HookOutcome


@dataclass
class RegexHook:
    """Pattern-matching hook with block, warn, or redact semantics.

    The same hook may run in ``before_call``, ``after_call``, or both phases.

    Note:
        ``action="redact"`` operates on raw text. When used on
        ``get_json_response`` / ``get_structured_json_response`` /
        ``get_json_schema_response``, redactions are not JSON-aware and can
        corrupt the response so that downstream parsing fails. Callers using
        redact on structured paths are responsible for choosing a
        ``redaction`` value that preserves JSON validity.
    """

    name: str
    pattern: str
    flags: int = 0
    phase: Literal["before", "after", "both"] = "both"
    action: Literal["block", "warn", "redact"] = "warn"
    redaction: str = "[REDACTED]"
    _compiled: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compiled = re.compile(self.pattern, self.flags)

    async def before_call(self, prompt: str, ctx: HookContext) -> HookOutcome:
        if self.phase == "after":
            return HookOutcome.pass_through(self.name)
        return self._evaluate(prompt)

    async def after_call(self, prompt: str, response: str, ctx: HookContext) -> HookOutcome:
        if self.phase == "before":
            return HookOutcome.pass_through(self.name)
        return self._evaluate(response)

    def _evaluate(self, text: str) -> HookOutcome:
        match = self._compiled.search(text)
        if match is None:
            return HookOutcome.pass_through(self.name)

        reason = f"matched pattern {self.pattern!r}"
        if self.action == "block":
            return HookOutcome.block(self.name, reason)
        if self.action == "warn":
            return HookOutcome.warn(self.name, reason)
        redacted = self._compiled.sub(self.redaction, text)
        return HookOutcome.redact(self.name, redacted, reason)
