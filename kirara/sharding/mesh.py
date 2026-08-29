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

"""JAX mesh construction."""

from __future__ import annotations

import math

import jax
import numpy as np
from jax.sharding import Mesh


def build_mesh(spec: Mesh | dict[str, int] | None) -> Mesh | None:
    """Build a JAX mesh from axis sizes using the current device set.

    Args:
        spec: Mesh mapping or specification.

    Raises:
        ValueError: If mesh axis sizes are not positive.
        ValueError: If the required devices don't match available JAX devices.

    Returns:
        The constructed JAX Mesh, or None if spec is None.
    """
    if spec is None or isinstance(spec, Mesh):
        return spec
    if not spec or any(size < 1 for size in spec.values()):
        raise ValueError("mesh axis sizes must be positive")
    required = math.prod(spec.values())
    devices = jax.devices()
    if required != len(devices):
        raise ValueError(
            f"mesh requests {required} devices but JAX has {len(devices)}"
        )
    device_mesh = np.asarray(devices, dtype=object).reshape(tuple(spec.values()))
    return Mesh(device_mesh, tuple(spec))


__all__ = ["build_mesh"]
