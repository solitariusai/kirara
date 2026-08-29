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

"""Unified model adapter data structures and protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import jax

from kirara.state.base import State
from kirara.types import Metadata, ModelDType


@jax.tree_util.register_pytree_node_class
@dataclass
class Batch:
    """Runtime batch shared by every model adapter."""

    input_ids: jax.Array | None = None
    positions: jax.Array | None = None
    slot_ids: jax.Array | None = None
    active_mask: jax.Array | None = None
    modalities: Metadata | None = None
    metadata: Metadata | None = None

    def tree_flatten(self):
        children = (
            self.input_ids,
            self.positions,
            self.slot_ids,
            self.active_mask,
            self.modalities,
            self.metadata,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del auxiliary
        return cls(*children)


@dataclass(frozen=True, slots=True)
class Capabilities:
    generate: bool = False
    encode: bool = False
    multimodal: bool = False
    stateful: bool = False


@dataclass(frozen=True, slots=True)
class ModelConfig:
    num_layers: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    max_model_len: int | None = None
    dtype: ModelDType | None = None


@dataclass(frozen=True, slots=True)
class ModelStateSpec:
    """Structural description of persistent model state."""

    kinds: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@jax.tree_util.register_pytree_node_class
@dataclass
class ModelOutput:
    logits: jax.Array | None = None
    embeddings: jax.Array | None = None
    state: State | None = None
    metadata: Metadata | None = None

    def tree_flatten(self):
        return (
            self.logits,
            self.embeddings,
            self.state,
            self.metadata,
        ), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del auxiliary
        return cls(*children)


class Adapter(Protocol):
    @property
    def config(self) -> ModelConfig: ...

    @property
    def capabilities(self) -> Capabilities: ...

    @property
    def state_spec(self) -> ModelStateSpec | None: ...

    def __call__(
        self,
        batch: Batch,
        state: State | None = None,
    ) -> ModelOutput: ...


__all__ = [
    "Adapter",
    "Batch",
    "Capabilities",
    "ModelConfig",
    "ModelOutput",
    "ModelStateSpec",
]
