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

"""Bounded JAX compilation and model execution."""

from __future__ import annotations

from collections.abc import Callable

import jax
from jax.sharding import Mesh

from kirara.attention import AttentionBackend
from kirara.models import Adapter, Batch, Output
from kirara.state import State, StateManager
from kirara.types import ExecutionPhase

type ModelStep = Callable[[Batch, State], Output]


class Runner:
    """Own bounded JAX executables around one unified model adapter call."""

    def __init__(
        self,
        model: Adapter,
        state_manager: StateManager,
        attention_backend: AttentionBackend,
        mesh: Mesh | None,
        prefill_buckets: tuple[int, ...],
    ) -> None:
        """Initialize the runner with a model and state.

        Args:
            model (Adapter): The model adapter to execute.
            state_manager (StateManager): The state manager for tracking KV cache and other state.
            attention_backend (AttentionBackend): The attention backend to use.
            mesh (Mesh | None): The JAX mesh to shard over, or None for no sharding.
            prefill_buckets (tuple[int, ...]): Allowed prompt lengths for prefill batching.

        Raises:
            ValueError: If prefill_buckets is empty or contains non-positive lengths.
        """
        self.model = model
        self.state_manager = state_manager
        self.attention_backend = attention_backend
        self.mesh = mesh
        self.prefill_buckets = tuple(sorted(set(prefill_buckets)))
        if not self.prefill_buckets or any(
            bucket < 1 for bucket in self.prefill_buckets
        ):
            raise ValueError("prefill buckets must be positive")

        self._prefill_steps: dict[int, ModelStep] = {}
        self._decode_step = jax.jit(self._call_model)
        self._encode_steps: dict[int, ModelStep] = {}
        self.compiled_prefill_buckets: set[int] = set()
        self.prefill_bucket_history: list[int] = []
        self.prefill_model_invocations = 0
        self.decode_model_invocations = 0
        self.encode_model_invocations = 0
        self.decode_compiled = False

    def bucket_for_length(self, length: int) -> int:
        """Find the smallest prefill bucket that can fit the given length.

        Args:
            length (int): The sequence length to bucket.

        Raises:
            ValueError: If length is not positive.
            ValueError: If length exceeds the largest configured bucket.

        Returns:
            int: The size of the matched bucket.
        """
        if length < 1:
            raise ValueError("input length must be positive")
        for bucket in self.prefill_buckets:
            if length <= bucket:
                return bucket
        raise ValueError(
            f"input length {length} exceeds largest bucket "
            f"{self.prefill_buckets[-1]}"
        )

    @property
    def compiled_executable_count(self) -> int:
        """Get the total number of JAX executables compiled so far.

        Returns:
            int: The count of compiled prefill, decode, and encode steps.
        """
        return (
            len(self.compiled_prefill_buckets)
            + int(self.decode_compiled)
            + len(self._encode_steps)
        )

    def execute(self, batch: Batch, *, phase: ExecutionPhase) -> Output:
        """Execute a model step for the given batch and phase.

        Args:
            batch (Batch): The batched inputs and metadata.
            phase (ExecutionPhase): The phase of execution ('prefill', 'decode', 'encode').

        Raises:
            ValueError: If an unknown execution phase is provided.

        Returns:
            Output: The model outputs including logits and updated state.
        """
        state = self.state_manager.state
        if phase == "prefill":
            bucket = batch.input_ids.shape[1]
            step = self._prefill_steps.get(bucket)
            if step is None:
                step = jax.jit(self._call_model)
                self._prefill_steps[bucket] = step
            output = step(batch, state)
            self.compiled_prefill_buckets.add(bucket)
            self.prefill_bucket_history.append(bucket)
            self.prefill_model_invocations += 1
        elif phase == "decode":
            output = self._decode_step(batch, state)
            self.decode_compiled = True
            self.decode_model_invocations += 1
        elif phase == "encode":
            bucket = batch.input_ids.shape[1]
            step = self._encode_steps.get(bucket)
            if step is None:
                step = jax.jit(self._call_model)
                self._encode_steps[bucket] = step
            output = step(batch, state)
            self.encode_model_invocations += 1
        else:
            raise ValueError(f"unknown execution phase: {phase}")

        if output.state is not None:
            self.state_manager.update(output.state, batch.active_mask)
        return output

    def _call_model(self, batch: Batch, state: State) -> Output:
        """Call the underlying model adapter with the batch and state.

        Args:
            batch (Batch): The input batch to process.
            state (State): The current model state.

        Returns:
            Output: The resulting model output.
        """
        return self.model(batch, state)


__all__ = ["Runner"]
