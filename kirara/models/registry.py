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

"""Registry-driven model integration resolution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

from kirara.attention import AttentionBackend
from kirara.inputs import InputProcessor
from kirara.models.base import Adapter
from kirara.models.loader import LoadedModel, Loader
from kirara.types import ModelSource


class AdapterRuntime(TypedDict):
    block_size: int


type AdapterFactory = Callable[
    [LoadedModel, AttentionBackend, AdapterRuntime],
    Adapter,
]
type ProcessorFactory = Callable[[LoadedModel], InputProcessor]
type ModelDetector = Callable[[ModelSource], bool]


@dataclass(frozen=True, slots=True)
class ModelIntegration:
    """Integration definition for a specific model architecture."""

    loader: Loader
    adapter: AdapterFactory
    processor: ProcessorFactory
    detect: ModelDetector
    default_for_repositories: bool = False


class ModelRegistry:
    """Resolve model integrations without branches in the public API."""

    def __init__(self) -> None:
        self._integrations: dict[str, ModelIntegration] = {}

    def register(
        self,
        name: str,
        *,
        loader: Loader,
        adapter: AdapterFactory,
        processor: ProcessorFactory,
        detect: ModelDetector,
        default_for_repositories: bool = False,
    ) -> None:
        """Register a new model integration.

        Args:
            name: Unique name for the model integration.
            loader: Model loading implementation.
            adapter: Factory for creating the model adapter.
            processor: Factory for creating the input processor.
            detect: Function to detect if a source is supported.
            default_for_repositories: Whether this is the default fallback.

        Raises:
            ValueError: If the integration name is empty.
        """
        if not name:
            raise ValueError("integration name cannot be empty")
        self._integrations[name] = ModelIntegration(
            loader=loader,
            adapter=adapter,
            processor=processor,
            detect=detect,
            default_for_repositories=default_for_repositories,
        )

    def resolve(
        self,
        source: ModelSource,
        model_impl: str = "auto",
    ) -> ModelIntegration:
        """Resolve a model integration for a given source.

        Args:
            source: Source identifier for the model.
            model_impl: Specific implementation name, or "auto" to detect.

        Raises:
            ValueError: If the specified implementation is unknown.
            ValueError: If no registered integration accepts the source.

        Returns:
            ModelIntegration: The resolved model integration.
        """
        if model_impl != "auto":
            try:
                return self._integrations[model_impl]
            except KeyError as error:
                available = ", ".join(sorted(self._integrations))
                raise ValueError(
                    f"unknown model implementation {model_impl!r}; "
                    f"available: {available}"
                ) from error

        for integration in self._integrations.values():
            if integration.detect(source):
                return integration
        if isinstance(source, str):
            for integration in self._integrations.values():
                if integration.default_for_repositories:
                    return integration
        raise ValueError("no registered integration accepts this model source")

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._integrations))


model_registry = ModelRegistry()


__all__ = [
    "AdapterFactory",
    "AdapterRuntime",
    "ModelDetector",
    "ModelIntegration",
    "ModelRegistry",
    "ProcessorFactory",
    "model_registry",
]
