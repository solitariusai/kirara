from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Generation settings shared by native and external model paths."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    eos_token_ids: int | list[int] | None = None
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True, slots=True)
class OutputSampling:
    """One generated continuation for a prompt."""

    text: str
    token_ids: list[int]
    num_tokens: int
    stop_reason: str


@dataclass(frozen=True, slots=True)
class Output:
    """A prompt and all continuations sampled for it."""

    prompt: str
    prompt_ids: list[int]
    output: list[OutputSampling]


__all__ = ["Output", "OutputSampling", "SamplingParams"]
