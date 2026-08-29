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

"""Persistent state allocation, updates, and recycling."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from kirara.state.base import State
from kirara.state.paged import PagedCacheManager, PagedKVState
from kirara.state.prefix import PrefixCache


class StateManager:
    """Own host allocation metadata and authoritative device model state."""

    def __init__(
        self,
        paged_cache: PagedCacheManager | None = None,
        slot_state: dict[str, jax.Array] | None = None,
        enable_prefix_caching: bool = False,
        **persistent_state: Any,
    ) -> None:
        """Initialize the state manager.

        Args:
            paged_cache (PagedCacheManager | None, optional): The paged cache manager for KV state. Defaults to None.
            slot_state: Static per-request recurrent or hybrid state arrays.
            enable_prefix_caching: Whether completed KV pages may be retained.
            **persistent_state (Any): Additional persistent state values.
        """
        self.paged_cache = paged_cache
        if enable_prefix_caching and paged_cache is None:
            raise ValueError("prefix caching requires paged KV state")
        self.prefix_cache = (
            PrefixCache(paged_cache)
            if enable_prefix_caching and paged_cache is not None
            else None
        )
        self._slot_state_keys = set(slot_state or {})
        self._persistent_state = {
            **dict(persistent_state),
            **dict(slot_state or {}),
        }

    @property
    def state(self) -> State:
        """Get the complete model state, including paged KV state if available.

        Returns:
            State: The current model state.
        """
        values = dict(self._persistent_state)
        if self.paged_cache is not None:
            values["kv"] = self.paged_cache.state
        return State(**values)

    def update(
        self,
        state: State,
        active_mask: jax.Array | None = None,
    ) -> None:
        """Update the persistent state with new values.

        Args:
            state (State): The new state containing updated values.

        Raises:
            TypeError: If the 'kv' state is present but is not a PagedKVState.
        """
        if self.paged_cache is not None and "kv" in state:
            kv_state = state["kv"]
            if not isinstance(kv_state, PagedKVState):
                raise TypeError("state.kv must be PagedKVState")
            self.paged_cache.update_state(kv_state)
        for key, value in state.items():
            if key == "kv":
                continue
            if key not in self._slot_state_keys or active_mask is None:
                self._persistent_state[key] = value
                continue
            previous = self._persistent_state[key]
            row_active = jnp.asarray(active_mask, dtype=jnp.bool_)
            if row_active.ndim > 1:
                row_active = jnp.any(
                    row_active,
                    axis=tuple(range(1, row_active.ndim)),
                )
            if value.shape != previous.shape:
                raise ValueError(
                    f"slot state {key!r} changed shape from "
                    f"{previous.shape} to {value.shape}"
                )
            if row_active.shape[0] < value.shape[0]:
                row_active = jnp.pad(
                    row_active,
                    (0, value.shape[0] - row_active.shape[0]),
                )
            elif row_active.shape[0] != value.shape[0]:
                raise ValueError("active mask does not match slot state rows")
            broadcast = row_active.reshape(
                (row_active.shape[0],)
                + (1,) * (value.ndim - 1)
            )
            self._persistent_state[key] = jnp.where(
                broadcast,
                value,
                previous,
            )

    def ensure_position(self, slot_id: int, position: int) -> int:
        """Ensure a physical block is allocated for the given position in a request slot.

        Args:
            slot_id (int): The request slot ID.
            position (int): The sequence position to allocate.

        Raises:
            RuntimeError: If the paged state runs out of physical blocks.

        Returns:
            int: The physical block index.
        """
        if self.paged_cache is None:
            return -1
        while True:
            try:
                return self.paged_cache.ensure_position(slot_id, position)
            except RuntimeError:
                if self.prefix_cache is None or not self.prefix_cache.evict_one():
                    raise

    def ensure_length(self, slot_id: int, sequence_length: int) -> None:
        """Ensure physical blocks are allocated for the given sequence length.

        Args:
            slot_id (int): The request slot ID.
            sequence_length (int): The target sequence length.

        Raises:
            RuntimeError: If the paged state runs out of physical blocks.
        """
        if self.paged_cache is None:
            return
        self.ensure_position(slot_id, sequence_length - 1)

    def set_sequence_length(self, slot_id: int, sequence_length: int) -> None:
        """Set the current sequence length for a given slot.

        Args:
            slot_id (int): The request slot ID.
            sequence_length (int): The sequence length to set.

        Raises:
            RuntimeError: If the model has no paged state.
        """
        if self.paged_cache is None:
            return
        self.paged_cache.set_sequence_length(slot_id, sequence_length)

    def reset_slot(self, slot_id: int) -> None:
        """Reset and release all physical blocks for a given slot.

        Args:
            slot_id (int): The request slot ID to reset.
        """
        if self.paged_cache is not None:
            self.paged_cache.reset_slot(slot_id)
        for key in self._slot_state_keys:
            value = self._persistent_state[key]
            self._persistent_state[key] = value.at[slot_id].set(
                jnp.zeros(value.shape[1:], dtype=value.dtype)
            )

    def restore_prefix(self, slot_id: int, tokens: list[int]) -> int:
        """Attach the longest reusable prefix while leaving a token for logits."""
        if self.prefix_cache is None or self.paged_cache is None:
            return 0
        max_reusable = (
            (len(tokens) - 1) // self.paged_cache.block_size
        ) * self.paged_cache.block_size
        entry = self.prefix_cache.lookup(tokens, max_reusable)
        if entry is None:
            return 0
        self.paged_cache.attach_blocks(slot_id, entry.blocks)
        self.paged_cache.set_sequence_length(slot_id, len(entry.tokens))
        return len(entry.tokens)

    def publish_prefix(
        self,
        slot_id: int,
        tokens: list[int],
        processed_length: int,
    ) -> None:
        """Publish completed full pages without caching a prompt's last token."""
        if self.prefix_cache is None or self.paged_cache is None:
            return
        cacheable = min(processed_length, len(tokens) - 1)
        cacheable -= cacheable % self.paged_cache.block_size
        if cacheable <= 0:
            return
        block_count = cacheable // self.paged_cache.block_size
        blocks = self.paged_cache.slot_blocks(slot_id)[:block_count]
        self.prefix_cache.insert(tuple(tokens[:cacheable]), blocks)


__all__ = ["StateManager"]
