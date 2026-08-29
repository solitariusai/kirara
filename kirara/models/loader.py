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

"""Model loading contracts and loaded model metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from jax.sharding import Mesh

from kirara.types import Metadata, ModelDType, ModelKwargs, ModelSource


@dataclass(slots=True)
class LoadedModel:
    model: Any
    tokenizer: Any = None
    processor: Any = None
    metadata: Metadata | None = None


class ModelLoader(Protocol):
    def load(
        self,
        source: ModelSource,
        *,
        dtype: ModelDType,
        revision: str | None,
        tokenizer: str | None,
        quantization: Any,
        mesh: Mesh | None,
        trust_remote_code: bool,
        model_kwargs: ModelKwargs,
    ) -> LoadedModel: ...


__all__ = ["LoadedModel", "ModelLoader"]
