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

"""Authoritative paged KV state and physical block allocator."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass
class PagedKVState:
    key_pool: jax.Array
    value_pool: jax.Array
    block_table: jax.Array
    sequence_lengths: jax.Array

    def tree_flatten(self):
        return (
            self.key_pool,
            self.value_pool,
            self.block_table,
            self.sequence_lengths,
        ), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del auxiliary
        return cls(*children)


class PagedCacheManager:
    """Host-side block allocator plus one authoritative device KV pool."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: Any = jnp.float32,
        block_size: int = 16,
        num_blocks: int | None = None,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = jnp.dtype(dtype)
        self.block_size = int(block_size)
        self.max_blocks_per_sequence = math.ceil(max_seq_len / block_size)
        if num_blocks is None:
            num_blocks = max_batch_size * self.max_blocks_per_sequence
        if num_blocks < max_batch_size:
            raise ValueError("num_blocks must be at least max_batch_size")
        self.num_blocks = int(num_blocks)

        pool_shape = (
            num_layers,
            self.num_blocks,
            self.block_size,
            num_heads,
            head_dim,
        )
        self.cache = (
            jnp.zeros(pool_shape, dtype=self.dtype),
            jnp.zeros(pool_shape, dtype=self.dtype),
        )
        self.block_table = jnp.full(
            (max_batch_size, self.max_blocks_per_sequence),
            -1,
            dtype=jnp.int32,
        )
        self.sequence_lengths = jnp.zeros(
            (max_batch_size,),
            dtype=jnp.int32,
        )
        self.allocated_block_count = jnp.zeros(
            (max_batch_size,),
            dtype=jnp.int32,
        )
        self._slot_blocks: list[list[int]] = [
            [] for _ in range(max_batch_size)
        ]
        self._free_blocks = list(range(self.num_blocks))
        heapq.heapify(self._free_blocks)

    @property
    def free_block_count(self) -> int:
        return len(self._free_blocks)

    def slot_blocks(self, slot_id: int) -> tuple[int, ...]:
        return tuple(self._slot_blocks[slot_id])

    @property
    def state(self) -> PagedKVState:
        return PagedKVState(
            key_pool=self.cache[0],
            value_pool=self.cache[1],
            block_table=self.block_table,
            sequence_lengths=self.sequence_lengths,
        )

    def update_state(self, state: PagedKVState) -> None:
        self.cache = (state.key_pool, state.value_pool)
        self.block_table = state.block_table
        self.sequence_lengths = state.sequence_lengths

    def ensure_position(self, slot_id: int, position: int) -> int:
        """Allocate the logical block containing ``position`` if necessary."""
        if position < 0 or position >= self.max_seq_len:
            raise ValueError(
                f"cache position {position} is outside [0, {self.max_seq_len})"
            )
        logical_block = position // self.block_size
        blocks = self._slot_blocks[slot_id]
        while len(blocks) <= logical_block:
            if not self._free_blocks:
                raise RuntimeError("paged KV cache is out of physical blocks")
            physical_block = heapq.heappop(self._free_blocks)
            blocks.append(physical_block)
            table_index = len(blocks) - 1
            self.block_table = self.block_table.at[
                slot_id, table_index
            ].set(physical_block)
        self.allocated_block_count = self.allocated_block_count.at[
            slot_id
        ].set(len(blocks))
        return blocks[logical_block]

    def ensure_length(self, slot_id: int, sequence_length: int) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.ensure_position(slot_id, sequence_length - 1)

    def set_sequence_length(self, slot_id: int, sequence_length: int) -> None:
        if sequence_length < 0 or sequence_length > self.max_seq_len:
            raise ValueError("sequence length exceeds cache capacity")
        self.sequence_lengths = self.sequence_lengths.at[slot_id].set(
            sequence_length
        )

    def reset_slot(self, slot_id: int) -> None:
        """Zero and release every physical block owned by one request."""
        blocks = self._slot_blocks[slot_id]
        if blocks:
            block_ids = jnp.asarray(blocks, dtype=jnp.int32)
            self.cache = (
                self.cache[0].at[:, block_ids].set(0),
                self.cache[1].at[:, block_ids].set(0),
            )
            for block_id in blocks:
                heapq.heappush(self._free_blocks, block_id)
        self._slot_blocks[slot_id] = []
        self.block_table = self.block_table.at[slot_id].set(-1)
        self.sequence_lengths = self.sequence_lengths.at[slot_id].set(0)
        self.allocated_block_count = self.allocated_block_count.at[
            slot_id
        ].set(0)


# Source-compatible name for callers that imported CacheManager.
CacheManager = PagedCacheManager


def write_paged_kv(
    pool: jax.Array,
    updates: jax.Array,
    block_table: jax.Array,
    positions: jax.Array,
    token_active: jax.Array,
    block_size: int,
) -> jax.Array:
    """Scatter ``[batch, query, heads, dim]`` updates into a page pool."""
    batch_size, query_length = positions.shape
    flat_updates = updates.reshape(
        batch_size * query_length,
        *updates.shape[2:],
    )
    flat_positions = positions.reshape(-1)
    flat_active = token_active.reshape(-1)
    flat_slots = jnp.repeat(
        jnp.arange(batch_size, dtype=jnp.int32),
        query_length,
    )

    def write_one(index, current_pool):
        position = flat_positions[index]
        logical_block = position // block_size
        safe_logical_block = jnp.clip(
            logical_block,
            0,
            block_table.shape[1] - 1,
        )
        physical_block = block_table[
            flat_slots[index], safe_logical_block
        ]
        valid = flat_active[index] & (physical_block >= 0)

        def do_write(value):
            return value.at[
                physical_block,
                position % block_size,
            ].set(flat_updates[index])

        return jax.lax.cond(valid, do_write, lambda value: value, current_pool)

    return jax.lax.fori_loop(
        0,
        batch_size * query_length,
        write_one,
        pool,
    )


__all__ = [
    "CacheManager",
    "PagedCacheManager",
    "PagedKVState",
    "write_paged_kv",
]
