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

"""Registration of built-in model integrations."""

from kirara.integrations.custom import register as register_custom
from kirara.integrations.taktiny import register as register_taktiny
from kirara.models.registry import ModelRegistry


def register_builtin_integrations(registry: ModelRegistry) -> None:
    """Registers all built-in model integrations with the given registry.

    Args:
        registry (ModelRegistry): The model registry to register integrations with.
    """
    if "custom" not in registry.names():
        register_custom(registry)
    if "taktiny" not in registry.names():
        register_taktiny(registry)


__all__ = ["register_builtin_integrations"]
