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

"""Paged causal attention over Kirara's authoritative KV state."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from kirara.attention.base import AttentionMetadata
from kirara.state import State


def paged_attention(
    query: jax.Array,
    key_pool: jax.Array,
    value_pool: jax.Array,
    block_table: jax.Array,
    sequence_lengths: jax.Array,
    query_positions: jax.Array,
    query_active: jax.Array,
    *,
    block_size: int,
    scale: float | None = None,
    softcap: float | None = None,
) -> jax.Array:
    """Causal MHA/MQA/GQA over page tiles with a global score softmax.

    Only one physical block is gathered per loop iteration.  A contiguous
    per-request KV sequence is never constructed.
    """
    batch_size, query_length, query_heads, head_dim = query.shape
    kv_heads = key_pool.shape[2]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    groups = query_heads // kv_heads
    if value_pool.shape != key_pool.shape:
        raise ValueError("key and value pools must have identical shapes")
    if key_pool.shape[1] != block_size:
        raise ValueError("pool block dimension does not match block_size")
    if scale is None:
        scale = head_dim**-0.5

    q = query.reshape(
        batch_size,
        query_length,
        kv_heads,
        groups,
        head_dim,
    )
    scores_by_block = jnp.zeros(
        (
            batch_size,
            kv_heads,
            groups,
            query_length,
            block_table.shape[1],
            block_size,
        ),
        dtype=jnp.float32,
    )
    block_offsets = jnp.arange(block_size, dtype=jnp.int32)
    mask_value = jnp.asarray(-1.0e30, dtype=jnp.float32)

    def score_block(logical_block, all_scores):
        physical_ids = block_table[:, logical_block]
        safe_ids = jnp.maximum(physical_ids, 0)
        block_keys = key_pool[safe_ids]
        scores = jnp.einsum(
            "bqhgd,bkhd->bhgqk",
            q,
            block_keys,
            preferred_element_type=jnp.float32,
        ) * scale
        if softcap is not None:
            scores = softcap * jnp.tanh(scores / softcap)

        key_positions = logical_block * block_size + block_offsets
        valid_keys = key_positions[None, :] < sequence_lengths[:, None]
        causal = (
            key_positions[None, None, :]
            <= query_positions[:, :, None]
        )
        valid = (
            valid_keys[:, None, :]
            & causal
            & query_active[:, :, None]
            & (physical_ids >= 0)[:, None, None]
        )
        scores = jnp.where(valid[:, None, None], scores, mask_value)
        return all_scores.at[..., logical_block, :].set(scores)

    scores_by_block = jax.lax.fori_loop(
        0,
        block_table.shape[1],
        score_block,
        scores_by_block,
    )
    probabilities = jax.nn.softmax(
        scores_by_block.reshape(
            batch_size,
            kv_heads,
            groups,
            query_length,
            block_table.shape[1] * block_size,
        ),
        axis=-1,
    ).astype(key_pool.dtype)
    probabilities = probabilities.reshape(scores_by_block.shape)
    accumulator = jnp.zeros(
        (batch_size, kv_heads, groups, query_length, head_dim),
        dtype=query.dtype,
    )

    def apply_value_block(logical_block, current_accumulator):
        physical_ids = block_table[:, logical_block]
        safe_ids = jnp.maximum(physical_ids, 0)
        block_values = value_pool[safe_ids]
        block_output = jnp.einsum(
            "bhgqk,bkhd->bhgqd",
            probabilities[..., logical_block, :],
            block_values,
        )
        return current_accumulator + block_output

    accumulator = jax.lax.fori_loop(
        0,
        block_table.shape[1],
        apply_value_block,
        accumulator,
    )
    output = accumulator.transpose(0, 3, 1, 2, 4).reshape(
        batch_size,
        query_length,
        query_heads,
        head_dim,
    )
    output = jnp.where(
        query_active[:, :, None, None],
        output,
        0,
    )
    return output.astype(query.dtype)


class PagedAttention:
    """Initial Kirara attention backend over authoritative paged KV state."""

    def __init__(self, block_size: int, **options: Any) -> None:
        self.block_size = block_size
        self.options = dict(options)

    def __call__(
        self,
        query: jax.Array,
        state: State,
        metadata: AttentionMetadata,
    ) -> jax.Array:
        kv = state.kv
        return paged_attention(
            query,
            kv.key_pool,
            kv.value_pool,
            kv.block_table,
            kv.sequence_lengths,
            metadata["query_positions"],
            metadata["query_active"],
            block_size=self.block_size,
            scale=metadata.get("scale"),
            softcap=metadata.get("softcap"),
        )


__all__ = ["PagedAttention", "paged_attention"]
