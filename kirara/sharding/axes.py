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

"""Logical tensor-axis to mesh-axis mappings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AxisRules:
    """Ordered logical-axis to mesh-axis mappings."""

    rules: tuple[tuple[str, str | None], ...] = ()

    def mesh_axis(self, logical_axis: str) -> str | None:
        """Get the corresponding mesh axis for a logical axis.

        Args:
            logical_axis: The name of the logical axis.

        Returns:
            The mesh axis name, or None if not mapped.
        """
        for name, mesh_axis in self.rules:
            if name == logical_axis:
                return mesh_axis
        return None


__all__ = ["AxisRules"]
