from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kirara.api.schemas import SamplingParams

type ExternalGenerate = Callable[
    [list[list[int]], SamplingParams],
    list[list[int]],
]


class XLLM:
    """Adapt an external LLM's generator to Kirara's token interface.

    By default, the wrapped object's ``generate`` method is used. It must
    accept ``(input_ids, sampling_params)`` and return continuation token IDs
    as ``list[list[int]]``. Pass ``generate_fn`` when an external library uses
    a different calling convention.
    """

    def __init__(
        self,
        llm: Any,
        generate_fn: ExternalGenerate | None = None,
    ) -> None:
        self.llm = llm
        generator = generate_fn or getattr(llm, "generate", None)
        if not callable(generator):
            raise TypeError(
                "external LLM must provide generate or an explicit generate_fn"
            )
        self._generate = generator

    def generate(
        self,
        input_ids: list[list[int]],
        sampling_params: SamplingParams,
    ) -> list[list[int]]:
        """Delegate generation without using Kirara scheduler or kernels."""
        generated = self._generate(input_ids, sampling_params)
        return self._validate_output(generated, len(input_ids))

    @staticmethod
    def _validate_output(
        generated: Any,
        expected_rows: int,
    ) -> list[list[int]]:
        if hasattr(generated, "tolist"):
            generated = generated.tolist()
        if not isinstance(generated, list) or len(generated) != expected_rows:
            raise TypeError(
                "external generator must return one token list per prompt"
            )
        if not all(
            isinstance(row, list)
            and all(isinstance(token, int) for token in row)
            for row in generated
        ):
            raise TypeError(
                "external generator must return list[list[int]]"
            )
        return [list(row) for row in generated]


__all__ = ["ExternalGenerate", "XLLM"]
