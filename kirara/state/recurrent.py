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

"""Persistent recurrent state representation."""

from __future__ import annotations

from dataclasses import dataclass

import jax


@jax.tree_util.register_pytree_node_class
@dataclass
class RecurrentState:
    """Persistent state representation for recurrent or convolution layers."""
    
    recurrent: jax.Array | None = None
    convolution: jax.Array | None = None

    def tree_flatten(self):
        return (self.recurrent, self.convolution), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del auxiliary
        return cls(*children)


__all__ = ["RecurrentState"]
