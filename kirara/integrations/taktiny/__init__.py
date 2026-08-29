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

"""TakTiny integration registration."""

from __future__ import annotations

from kirara.attention import AttentionBackend
from kirara.integrations.taktiny.adapter import TakTinyAdapter
from kirara.integrations.taktiny.loader import TakTinyLoader
from kirara.integrations.taktiny.processor import create_processor
from kirara.models.loader import LoadedModel
from kirara.models.registry import AdapterRuntime, ModelRegistry
from kirara.types import ModelSource


def _detect(source: ModelSource) -> bool:
    if isinstance(source, str):
        return False
    return hasattr(source, "paged_forward") or hasattr(source, "model")


def _create_adapter(
    loaded: LoadedModel,
    attention_backend: AttentionBackend,
    runtime: AdapterRuntime,
) -> TakTinyAdapter:
    return TakTinyAdapter(
        loaded.model,
        attention_backend,
        block_size=runtime["block_size"],
    )


def register(registry: ModelRegistry) -> None:
    registry.register(
        "taktiny",
        loader=TakTinyLoader(),
        adapter=_create_adapter,
        processor=create_processor,
        detect=_detect,
        default_for_repositories=True,
    )


__all__ = ["TakTinyAdapter", "TakTinyLoader", "register"]
