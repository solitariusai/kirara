# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text and structured multimodal input normalization."""

from __future__ import annotations

from typing import Any, Protocol

from kirara.inputs.types import GenerationInput, InputPart, NormalizedInput


class InputProcessor(Protocol):
    def __call__(self, inputs: GenerationInput) -> list[NormalizedInput]: ...

    def decode(self, token_ids: list[int]) -> str: ...


class TokenizerInputProcessor:
    """Tokenizer-backed text processor with structured modality passthrough."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, inputs: GenerationInput) -> list[NormalizedInput]:
        rows = self._rows(inputs)
        return [self._normalize(row) for row in rows]

    def decode(self, token_ids: list[int]) -> str:
        decode = getattr(self.tokenizer, "decode", None)
        if not callable(decode):
            return ""
        return str(decode(token_ids))

    def _normalize(self, row: str | list[InputPart]) -> NormalizedInput:
        if isinstance(row, str):
            return NormalizedInput(
                input_ids=self._encode(row),
                prompt=row,
            )

        text_parts: list[str] = []
        modalities: dict[str, list[Any]] = {}
        for part in row:
            part_type = part.get("type")
            if part_type == "text":
                text_parts.append(str(part.get("text", "")))
                continue
            if not isinstance(part_type, str):
                raise ValueError("input parts require a string type")
            value = part.get(part_type)
            modalities.setdefault(part_type, []).append(value)
        text = "".join(text_parts)
        return NormalizedInput(
            input_ids=self._encode(text),
            prompt=text,
            modalities=modalities,
            metadata={"parts": row},
        )

    def _encode(self, text: str) -> list[int]:
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            result = encode(text)
        elif callable(self.tokenizer):
            result = self.tokenizer(text)
        else:
            raise TypeError("tokenizer must be callable or expose encode")
        return _extract_token_ids(result)

    @staticmethod
    def _rows(
        inputs: GenerationInput,
    ) -> list[str | list[InputPart]]:
        if isinstance(inputs, str):
            return [inputs]
        if not isinstance(inputs, list):
            raise TypeError("inputs must be text or structured input parts")
        if not inputs:
            return []
        if all(isinstance(row, str) for row in inputs):
            return list(inputs)
        if all(isinstance(part, dict) for part in inputs):
            return [list(inputs)]
        if all(
            isinstance(row, list)
            and all(isinstance(part, dict) for part in row)
            for row in inputs
        ):
            return [list(row) for row in inputs]
        raise TypeError("structured inputs must be list[list[dict]]")


def _extract_token_ids(result: Any) -> list[int]:
    input_ids = getattr(result, "input_ids", None)
    if input_ids is None and isinstance(result, dict):
        input_ids = result.get("input_ids")
    if input_ids is None:
        input_ids = result
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if (
        isinstance(input_ids, list)
        and len(input_ids) == 1
        and isinstance(input_ids[0], list)
    ):
        input_ids = input_ids[0]
    if not isinstance(input_ids, list) or not all(
        isinstance(token, int) for token in input_ids
    ):
        raise TypeError("tokenizer must return a one-dimensional token list")
    if not input_ids:
        raise ValueError("processed inputs must contain at least one token")
    return list(input_ids)


__all__ = ["InputProcessor", "TokenizerInputProcessor"]
