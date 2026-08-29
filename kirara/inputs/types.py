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

"""Typed public and normalized input representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from kirara.types import Metadata


class _InputPayload(TypedDict, total=False):
    """Optional payload fields for an input part."""

    text: str
    image: Any
    audio: Any
    video: Any


class InputPart(_InputPayload):
    """A typed modality tag plus an integration-defined payload."""

    type: str


type PromptInput = str | list[InputPart]
type GenerationInput = PromptInput | list[str] | list[list[InputPart]]


@dataclass(slots=True)
class NormalizedInput:
    """Normalized generation input containing resolved tokens and extracted modalities."""
    
    input_ids: list[int]
    prompt: str = ""
    modalities: Metadata = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)


__all__ = [
    "GenerationInput",
    "InputPart",
    "NormalizedInput",
    "PromptInput",
]
