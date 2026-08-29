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

import jax.numpy as jnp

from kirara.inputs.types import GenerationInput, InputPart, NormalizedInput
from kirara.types import Metadata


class InputProcessor(Protocol):
    """Protocol for normalizing input text and modalities into token IDs."""

    def __call__(self, inputs: GenerationInput) -> list[NormalizedInput]:
        """Normalizes raw generation inputs into token sequences and metadata."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decodes a sequence of token IDs back into a string."""
        ...


class TokenizerInputProcessor:
    """Tokenizer-backed text processor with structured modality passthrough."""

    def __init__(self, tokenizer: Any) -> None:
        """Initializes the processor with a tokenizer.

        Args:
            tokenizer (Any): The underlying tokenizer object.
        """
        self.tokenizer = tokenizer

    def __call__(self, inputs: GenerationInput) -> list[NormalizedInput]:
        """Processes text or structured generation inputs into normalized representations.

        Args:
            inputs (GenerationInput): The raw text or structured inputs.

        Returns:
            list[NormalizedInput]: The normalized inputs ready for generation.
        """
        rows = self._rows(inputs)
        return [self._normalize(row) for row in rows]

    def decode(self, token_ids: list[int]) -> str:
        """Decodes token IDs to text using the underlying tokenizer.

        Args:
            token_ids (list[int]): Sequence of token IDs.

        Returns:
            str: Decoded text string.
        """
        decode = getattr(self.tokenizer, "decode", None)
        if not callable(decode):
            return ""
        return str(decode(token_ids))

    def _normalize(self, row: str | list[InputPart]) -> NormalizedInput:
        """Normalizes a single row of input into a NormalizedInput.

        Args:
            row (str | list[InputPart]): A single input row (string or parts).

        Raises:
            ValueError: If an input part lacks a string type.

        Returns:
            NormalizedInput: The normalized input object.
        """
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
        """Encodes text into a list of token IDs.

        Args:
            text (str): The text to encode.

        Raises:
            TypeError: If the tokenizer is not callable or lacks encode method.

        Returns:
            list[int]: The sequence of token IDs.
        """
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
        """Extracts rows from GenerationInput.

        Args:
            inputs (GenerationInput): The raw inputs.

        Raises:
            TypeError: If inputs is not properly formatted text or structured input.
            TypeError: If structured inputs are not list[list[dict]].

        Returns:
            list[str | list[InputPart]]: A uniform list of rows.
        """
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
    """Extracts a flat list of token IDs from tokenizer output.

    Args:
        result (Any): The output from the tokenizer.

    Raises:
        TypeError: If the output cannot be coerced to a flat 1D list of ints.
        ValueError: If the processed output is empty.

    Returns:
        list[int]: The token IDs.
    """
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


def collate_modalities(
    rows: list[tuple[int, NormalizedInput]],
    batch_size: int,
) -> tuple[Metadata | None, Metadata | None]:
    """Stack processor-produced modality tensors by immutable runtime slot.

    Every value for a modality key must have the same static shape and dtype.
    Missing rows are represented by zeros plus a runtime boolean mask.
    """
    keys = sorted({key for _, row in rows for key in row.modalities})
    if not keys:
        return None, None

    modalities: Metadata = {}
    masks: Metadata = {}
    for key in keys:
        present = [
            (slot_id, jnp.asarray(row.modalities[key]))
            for slot_id, row in rows
            if key in row.modalities
        ]
        template = present[0][1]
        if any(
            value.shape != template.shape or value.dtype != template.dtype
            for _, value in present[1:]
        ):
            raise ValueError(
                f"modality {key!r} must have one static shape and dtype"
            )
        batched = jnp.zeros(
            (batch_size, *template.shape),
            dtype=template.dtype,
        )
        active = jnp.zeros((batch_size,), dtype=jnp.bool_)
        for slot_id, value in present:
            batched = batched.at[slot_id].set(value)
            active = active.at[slot_id].set(True)
        modalities[key] = batched
        masks[key] = active
    return modalities, masks


__all__ = [
    "InputProcessor",
    "TokenizerInputProcessor",
    "collate_modalities",
]
