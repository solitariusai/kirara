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

"""Integration for models already implementing Kirara's adapter contract."""

from __future__ import annotations

from typing import Any

from kirara.attention import AttentionBackend
from kirara.inputs import InputProcessor, TokenizerInputProcessor
from kirara.models import Adapter
from kirara.models.loader import LoadedModel
from kirara.models.registry import AdapterRuntime, ModelRegistry
from kirara.types import ModelSource


class CustomLoader:
    def load(self, source: ModelSource, **kwargs: Any) -> LoadedModel:
        if isinstance(source, str):
            raise TypeError("custom integration requires a model instance")
        model_kwargs = kwargs.get("model_kwargs") or {}
        if model_kwargs:
            raise ValueError(
                "model_kwargs cannot modify an existing model instance"
            )
        return LoadedModel(
            model=source,
            tokenizer=getattr(source, "tokenizer", None),
            processor=getattr(source, "processor", None),
        )


def _detect(source: ModelSource) -> bool:
    if isinstance(source, str):
        return False
    required = ("config", "capabilities", "state_spec")
    return callable(source) and all(hasattr(source, name) for name in required)


def _adapter(
    loaded: LoadedModel,
    attention_backend: AttentionBackend,
    runtime: AdapterRuntime,
) -> Adapter:
    del attention_backend, runtime
    return loaded.model


def _processor(loaded: LoadedModel) -> InputProcessor:
    if loaded.processor is not None:
        return loaded.processor
    if loaded.tokenizer is not None:
        return TokenizerInputProcessor(loaded.tokenizer)
    raise ValueError("custom adapters must expose processor or tokenizer")


def register(registry: ModelRegistry) -> None:
    registry.register(
        "custom",
        loader=CustomLoader(),
        adapter=_adapter,
        processor=_processor,
        detect=_detect,
    )


__all__ = ["CustomLoader", "register"]
