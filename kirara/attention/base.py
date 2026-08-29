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

"""Structural contracts for interchangeable attention backends."""

from __future__ import annotations

from typing import Protocol, TypedDict

import jax

from kirara.state.base import State


class _AttentionOptions(TypedDict, total=False):
    scale: float | jax.Array | None
    softcap: float | jax.Array | None


class AttentionMetadata(_AttentionOptions):
    query_positions: jax.Array
    query_active: jax.Array


class AttentionBackend(Protocol):
    def __call__(
        self,
        query: jax.Array,
        state: State,
        metadata: AttentionMetadata,
    ) -> jax.Array: ...


__all__ = ["AttentionBackend", "AttentionMetadata"]
