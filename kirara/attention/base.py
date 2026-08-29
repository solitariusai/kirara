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
    """Optional configuration for attention backends.

    Args:
        scale (float | jax.Array | None): The attention scale factor.
        softcap (float | jax.Array | None): Softcapping threshold for attention logits.
    """
    scale: float | jax.Array | None
    softcap: float | jax.Array | None


class AttentionMetadata(_AttentionOptions):
    """Metadata required by the attention backend for a given execution step.

    Args:
        query_positions (jax.Array): The sequence positions of the query tokens.
        query_active (jax.Array): Boolean mask indicating active queries.
    """
    query_positions: jax.Array
    query_active: jax.Array


class AttentionBackend(Protocol):
    """Protocol defining the interface for attention backends."""
    def __call__(
        self,
        query: jax.Array,
        state: State,
        metadata: AttentionMetadata,
    ) -> jax.Array:
        """Compute attention for a query over the provided state.

        Args:
            query (jax.Array): The input query tensor.
            state (State): The model state containing KV cache.
            metadata (AttentionMetadata): Auxiliary information for the attention step.

        Returns:
            jax.Array: The attention output tensor.
        """
        ...


__all__ = ["AttentionBackend", "AttentionMetadata"]
