"""The small value types the API speaks.

Mirrors the shapes of ``typesafe_sdk`` (``Choice``, ``ChoiceAnswer``,
``SystemOneResponse``, ``Usage``) so code written against TypeSafe's SDK
keeps working, without taking the dependency: this package talks
to a remote vLLM, so it needs no torch and no transformers either.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Choice:
    """One multiple-choice question.

    ``criteria`` maps an option name to a description of when it applies;
    ``None`` means the name speaks for itself. Order is the index order.
    """

    criteria: Mapping[str, str | None]
    instructions: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.criteria, Mapping) or not self.criteria:
            raise ValueError("A Choice needs a nonempty criteria mapping")
        if not all(isinstance(name, str) and name for name in self.criteria):
            raise ValueError("Option names must be nonempty strings")
        if len(self.criteria) > 255:
            raise ValueError("A Choice takes at most 255 options")


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class Usage:
    """Summed over every request a call made, not per request.

    ``cached_tokens`` is the server's own count of prefix-cache hits. It
    matters more than it looks: a short prompt can report zero cached tokens
    even when it is sent twice, so "cold and warm agreed" means "the cache
    never engaged", not "prefill is free".
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


@dataclass(frozen=True)
class SystemOneResponse:
    model: str
    answers: dict[str, ChoiceAnswer]
    usage: Usage = field(default_factory=Usage)
    timings: dict[str, Any] = field(default_factory=dict)

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """TypeSafe's SDK calls this ``choices``; ``answers`` is the same map."""
        return self.answers
