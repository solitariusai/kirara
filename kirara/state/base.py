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

"""Mapping-compatible JAX pytree model state."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import jax


@jax.tree_util.register_pytree_node_class
class State(Mapping[str, Any]):
    """Named persistent model state with attribute and mapping access."""

    def __init__(self, **values: Any) -> None:
        self._values = dict(values)

    def __getattr__(self, name: str) -> Any:
        """Get an item from the state using attribute access.

        Args:
            name (str): The name of the state item to retrieve.

        Raises:
            AttributeError: If the requested state item does not exist.

        Returns:
            Any: The value of the requested state item.
        """
        try:
            return self._values[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __getitem__(self, name: str) -> Any:
        """Get an item from the state using mapping access.

        Args:
            name (str): The name of the state item to retrieve.

        Returns:
            Any: The value of the requested state item.
        """
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys in the state.

        Yields:
            Iterator[str]: An iterator over the state keys.
        """
        return iter(self._values)

    def __len__(self) -> int:
        """Get the number of items in the state.

        Returns:
            int: The number of state items.
        """
        return len(self._values)

    def tree_flatten(self):
        keys = tuple(self._values)
        values = tuple(self._values[key] for key in keys)
        return values, keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(**dict(zip(keys, values, strict=True)))


__all__ = ["State"]
