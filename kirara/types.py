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

"""Shared type vocabulary for Kirara's runtime boundaries.

Domain-specific structures remain with their owning packages. This module is
only for aliases shared across two or more boundaries; it is intentionally not
re-exported from :mod:`kirara`.
"""

from __future__ import annotations

from typing import Any, Literal

from jax.sharding import Mesh
from jax.typing import DTypeLike

type CacheDType = DTypeLike | Literal["auto"]
type ExecutionPhase = Literal["prefill", "decode", "encode"]
type MeshSpec = Mesh | dict[str, int] | None
type Metadata = dict[str, Any]
type ModelDType = DTypeLike
type ModelKwargs = dict[str, Any]
type ModelSource = object
type SchedulerPolicy = Literal["fcfs"]


__all__ = [
    "CacheDType",
    "ExecutionPhase",
    "MeshSpec",
    "Metadata",
    "ModelDType",
    "ModelKwargs",
    "ModelSource",
    "SchedulerPolicy",
]
