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

"""Request state and lifecycle metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from kirara.inputs import NormalizedInput
from kirara.sampling import SamplingParams


class RequestStatus(StrEnum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class Request:
    request_id: str
    inputs: NormalizedInput
    sampling_params: SamplingParams
    slot_id: int | None = None
    prompt_length: int = 0
    position: int = 0
    generated_tokens: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING

    def __post_init__(self) -> None:
        if not self.prompt_length:
            self.prompt_length = len(self.inputs.input_ids)


__all__ = ["Request", "RequestStatus"]
