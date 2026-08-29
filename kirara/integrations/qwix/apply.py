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

"""Preserve and validate Qwix rules before model-specific loading."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import qwix


def validate_qwix_config(config: Any) -> Any:
    """Validate a Qwix rule/provider and return the same object unchanged."""
    if config is None or isinstance(config, str):
        return config
    if isinstance(
        config,
        (qwix.QuantizationRule, qwix.PtqProvider),
    ):
        return config
    if isinstance(config, Sequence) and all(
        isinstance(rule, qwix.QuantizationRule) for rule in config
    ):
        return config
    raise TypeError(
        "quantization must be a Qwix QuantizationRule, provider, sequence "
        "of rules, or a TakTiny Qwix dtype shortcut"
    )


__all__ = ["validate_qwix_config"]
