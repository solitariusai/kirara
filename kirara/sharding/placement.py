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

"""NamedSharding construction from logical axes."""

from __future__ import annotations

from collections.abc import Sequence

from jax.sharding import Mesh, NamedSharding, PartitionSpec

from kirara.sharding.axes import AxisRules


def named_sharding(
    mesh: Mesh,
    logical_axes: Sequence[str | None],
    rules: AxisRules,
) -> NamedSharding:
    """Create a NamedSharding from logical axes and mapping rules.

    Args:
        mesh: The target device mesh.
        logical_axes: Sequence of logical axis names or None.
        rules: The rules mapping logical axes to mesh axes.

    Returns:
        The corresponding NamedSharding object.
    """
    partition = PartitionSpec(
        *(rules.mesh_axis(axis) if axis is not None else None for axis in logical_axes)
    )
    return NamedSharding(mesh, partition)


__all__ = ["named_sharding"]
