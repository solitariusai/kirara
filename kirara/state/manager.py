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

from kirara.state.base import State
from kirara.state.paged import PagedCacheManager, PagedKVState


class StateManager:
    """Own host allocation metadata and authoritative device model state."""

    def __init__(
        self,
        paged_cache: PagedCacheManager | None = None,
        **persistent_state: Any,
    ) -> None:
        """Initialize the state manager.

        Args:
            paged_cache (PagedCacheManager | None, optional): The paged cache manager for KV state. Defaults to None.
            **persistent_state (Any): Additional persistent state values.
        """
        self.paged_cache = paged_cache
        self._persistent_state = dict(persistent_state)

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

    def update(self, state: State) -> None:
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
        self._persistent_state = {
            key: value for key, value in state.items() if key != "kv"
        }

    def ensure_position(self, slot_id: int, position: int) -> int:
        """Ensure a physical block is allocated for the given position in a request slot.

        Args:
            slot_id (int): The request slot ID.
            position (int): The sequence position to allocate.

        Raises:
            RuntimeError: If the model has no paged state or runs out of physical blocks.

        Returns:
            int: The physical block index.
        """
        if self.paged_cache is None:
            raise RuntimeError("model has no paged state")
        return self.paged_cache.ensure_position(slot_id, position)

    def ensure_length(self, slot_id: int, sequence_length: int) -> None:
        """Ensure physical blocks are allocated for the given sequence length.

        Args:
            slot_id (int): The request slot ID.
            sequence_length (int): The target sequence length.

        Raises:
            RuntimeError: If the model has no paged state.
        """
        if self.paged_cache is None:
            raise RuntimeError("model has no paged state")
        self.paged_cache.ensure_length(slot_id, sequence_length)

    def set_sequence_length(self, slot_id: int, sequence_length: int) -> None:
        """Set the current sequence length for a given slot.

        Args:
            slot_id (int): The request slot ID.
            sequence_length (int): The sequence length to set.

        Raises:
            RuntimeError: If the model has no paged state.
        """
        if self.paged_cache is None:
            raise RuntimeError("model has no paged state")
        self.paged_cache.set_sequence_length(slot_id, sequence_length)

    def reset_slot(self, slot_id: int) -> None:
        """Reset and release all physical blocks for a given slot.

        Args:
            slot_id (int): The request slot ID to reset.
        """
        if self.paged_cache is not None:
            self.paged_cache.reset_slot(slot_id)


__all__ = ["StateManager"]
