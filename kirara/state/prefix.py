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

"""Reference-counted LRU cache of immutable paged-KV prompt prefixes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from kirara.state.paged import PagedCacheManager


@dataclass(frozen=True, slots=True)
class PrefixEntry:
    """Token identity and physical pages for one block-aligned prefix."""

    tokens: tuple[int, ...]
    blocks: tuple[int, ...]


class PrefixCache:
    """Hold bounded page references and return the longest matching prefix."""

    def __init__(self, cache: PagedCacheManager) -> None:
        self.cache = cache
        self._entries: OrderedDict[tuple[int, ...], PrefixEntry] = OrderedDict()
        self.max_cached_block_references = max(
            cache.num_blocks - cache.max_batch_size,
            0,
        )
        self._cached_block_references = 0
        self.hits = 0
        self.misses = 0

    def lookup(
        self,
        tokens: list[int],
        max_length: int,
    ) -> PrefixEntry | None:
        """Return the longest cached, block-aligned prefix within max_length."""
        token_tuple = tuple(tokens)
        best_key = None
        for key in self._entries:
            if len(key) <= max_length and token_tuple[: len(key)] == key:
                if best_key is None or len(key) > len(best_key):
                    best_key = key
        if best_key is None:
            self.misses += 1
            return None
        entry = self._entries.pop(best_key)
        self._entries[best_key] = entry
        self.hits += 1
        return entry

    def insert(
        self,
        tokens: tuple[int, ...],
        blocks: tuple[int, ...],
    ) -> None:
        """Retain an immutable prefix, evicting old entries to stay bounded."""
        if not tokens or tokens in self._entries:
            if tokens in self._entries:
                self._entries.move_to_end(tokens)
            return
        if len(blocks) > self.max_cached_block_references:
            return
        while (
            self._cached_block_references + len(blocks)
            > self.max_cached_block_references
        ):
            if not self.evict_one():
                return
        self.cache.retain_blocks(blocks)
        self._entries[tokens] = PrefixEntry(tokens, blocks)
        self._cached_block_references += len(blocks)

    def evict_one(self) -> bool:
        """Evict the least recently used prefix entry."""
        if not self._entries:
            return False
        _, entry = self._entries.popitem(last=False)
        self.cache.release_blocks(entry.blocks)
        self._cached_block_references -= len(entry.blocks)
        return True

    def clear(self) -> None:
        while self.evict_one():
            pass

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["PrefixCache", "PrefixEntry"]
