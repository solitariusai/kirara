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

"""Continuous batching and request lifecycle scheduling."""

from __future__ import annotations

from collections import deque
from typing import Any

import jax.numpy as jnp

from kirara.engine.request import PrefillChunk, Request, RequestStatus
from kirara.engine.runner import Runner
from kirara.engine.sampler import Sampler
from kirara.inputs import NormalizedInput, collate_modalities
from kirara.models import Batch
from kirara.sampling import SamplingParams
from kirara.state import StateManager
from kirara.types import SchedulerPolicy


class Scheduler:
    """Make request-level decisions without owning JAX compilation."""

    def __init__(
        self,
        runner: Runner,
        state_manager: StateManager,
        sampler: Sampler,
        *,
        max_num_seqs: int,
        max_model_len: int,
        max_num_batched_tokens: int | None = None,
        max_num_prefill_tokens: int | None = None,
        tokenizer: Any = None,
        policy: SchedulerPolicy = "fcfs",
        enable_chunked_prefill: bool = True,
        enable_prefix_caching: bool = False,
    ) -> None:
        """Initialize the scheduler for managing request lifecycles.

        Args:
            runner (Runner): The execution runner.
            state_manager (StateManager): State manager for tracking sequence contexts.
            sampler (Sampler): Token sampler.
            max_num_seqs (int): Maximum number of concurrent sequences.
            max_model_len (int): Maximum allowed sequence length.
            max_num_batched_tokens (int | None, optional): Maximum tokens per batch. Defaults to max_model_len.
            max_num_prefill_tokens (int | None, optional): Maximum tokens per prefill batch. Defaults to max_num_batched_tokens.
            tokenizer (Any, optional): Tokenizer for resolving stop tokens. Defaults to None.
            policy (SchedulerPolicy, optional): Scheduling policy. Defaults to "fcfs".
            enable_chunked_prefill (bool, optional): Whether to enable chunked prefill. Defaults to True.
            enable_prefix_caching (bool, optional): Whether to enable prefix caching. Defaults to False.

        Raises:
            ValueError: If policy is not 'fcfs'.
            ValueError: If max_num_batched_tokens is not positive.
            ValueError: If max_num_prefill_tokens is not positive.
        """
        if policy != "fcfs":
            raise ValueError("only fcfs scheduling is implemented")
        self.runner = runner
        self.state_manager = state_manager
        self.sampler = sampler
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = (
            max_model_len
            if max_num_batched_tokens is None
            else max_num_batched_tokens
        )
        self.max_num_prefill_tokens = (
            self.max_num_batched_tokens
            if max_num_prefill_tokens is None
            else max_num_prefill_tokens
        )
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_num_prefill_tokens < 1:
            raise ValueError("max_num_prefill_tokens must be positive")
        self.tokenizer = tokenizer
        self.policy = policy
        self.enable_chunked_prefill = enable_chunked_prefill
        self.enable_prefix_caching = enable_prefix_caching
        self.queue: deque[Request] = deque()
        self.slots: list[Request | None] = [None] * max_num_seqs
        self.next_request_id = 0
        self.prefill_token_history: list[int] = []

    @property
    def cache_manager(self):
        """Get the paged cache manager from the state manager.

        Returns:
            PagedCache: The cache manager instance.
        """
        return self.state_manager.paged_cache

    @property
    def prefill_model_invocations(self) -> int:
        """Get the total number of prefill model executions.

        Returns:
            int: The number of invocations.
        """
        return self.runner.prefill_model_invocations

    @property
    def prefill_bucket_history(self) -> list[int]:
        """Get the history of prefill bucket sizes used.

        Returns:
            list[int]: The list of bucket sizes.
        """
        return self.runner.prefill_bucket_history

    @property
    def compiled_prefill_buckets(self) -> set[int]:
        """Get the set of compiled prefill bucket sizes.

        Returns:
            set[int]: The compiled bucket sizes.
        """
        return self.runner.compiled_prefill_buckets

    @property
    def decode_model_invocations(self) -> int:
        """Get the total number of decode model executions.

        Returns:
            int: The number of invocations.
        """
        return self.runner.decode_model_invocations

    @property
    def decode_compiled(self) -> bool:
        """Check if the decode step has been compiled.

        Returns:
            bool: True if compiled, False otherwise.
        """
        return self.runner.decode_compiled

    @property
    def compiled_executable_count(self) -> int:
        """Get the total number of compiled JAX executables.

        Returns:
            int: The total count.
        """
        return self.runner.compiled_executable_count

    @property
    def _compiled_prefill_steps(self):
        """Get the cached prefill steps.

        Returns:
            dict: Mapping of bucket sizes to executable prefill steps.
        """
        return self.runner._prefill_steps

    def _get_bucket_for_len(self, length: int) -> int:
        """Get the appropriate prefill bucket size for a given sequence length.

        Args:
            length (int): The sequence length.

        Returns:
            int: The bucket size.
        """
        return self.runner.bucket_for_length(length)

    def add_request(
        self,
        inputs: NormalizedInput | list[int],
        sampling_params: SamplingParams | None = None,
    ) -> Request:
        """Add a new request to the scheduler queue.

        Args:
            inputs (NormalizedInput | list[int]): The input tokens or normalized input.
            sampling_params (SamplingParams | None, optional): The sampling configuration. Defaults to None.

        Returns:
            Request: The constructed request object.
        """
        normalized = (
            inputs
            if isinstance(inputs, NormalizedInput)
            else NormalizedInput(input_ids=list(inputs))
        )
        self._get_bucket_for_len(len(normalized.input_ids))
        request = Request(
            request_id=str(self.next_request_id),
            inputs=normalized,
            sampling_params=sampling_params or SamplingParams(),
        )
        self.queue.append(request)
        self.next_request_id += 1
        return request

    def step(self) -> bool:
        """Execute one scheduling step, running prefill or decode batches as needed.

        Returns:
            bool: True if there are still active or queued requests, False otherwise.
        """
        prefill_chunks = self._admit_prefill()
        prefill_tokens = sum(chunk.length for chunk in prefill_chunks)
        if prefill_chunks:
            self._run_batched_prefill(prefill_chunks)
            self._retire_finished(
                [chunk.request for chunk in prefill_chunks]
            )

        active_slots = [
            slot_id
            for slot_id, request in enumerate(self.slots)
            if request is not None
            and request.status is RequestStatus.DECODING
        ]
        decode_budget = max(
            self.max_num_batched_tokens - prefill_tokens,
            0,
        )
        active_slots = active_slots[:decode_budget]
        if active_slots:
            self._run_decode(active_slots)
            active_requests = [
                self.slots[slot_id]
                for slot_id in active_slots
                if self.slots[slot_id] is not None
            ]
            self._retire_finished(active_requests)

        return bool(self.queue) or any(self.slots)

    def generate(
        self,
        inputs: list[NormalizedInput] | list[list[int]],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[int]]:
        """Synchronously generate tokens for a batch of inputs until completion.

        Args:
            inputs (list[NormalizedInput] | list[list[int]]): The inputs to process.
            sampling_params (SamplingParams | None, optional): The sampling configuration. Defaults to None.

        Returns:
            list[list[int]]: The generated token sequences for each input.
        """
        requests = [
            self.add_request(item, sampling_params)
            for item in inputs
        ]
        while self.step():
            pass
        return [request.generated_tokens for request in requests]

    def _admit_prefill(self) -> list[PrefillChunk]:
        """Admit queued requests for the prefill phase based on budget.

        Raises:
            ValueError: If a prompt exceeds the token budget without chunked prefill enabled.
        Returns:
            list[PrefillChunk]: Prompt ranges scheduled for this step.
        """
        scheduled: list[PrefillChunk] = []
        total_tokens = 0
        prefill_budget = min(
            self.max_num_prefill_tokens,
            self.max_num_batched_tokens,
        )
        for request in self.slots:
            if request is None or request.status is not RequestStatus.PREFILL:
                continue
            remaining = request.prompt_length - request.prefill_position
            if remaining <= 0:
                continue
            chunk_length = min(remaining, prefill_budget - total_tokens)
            if chunk_length <= 0:
                break
            if chunk_length < remaining and not self.enable_chunked_prefill:
                break
            scheduled.append(
                PrefillChunk(
                    request,
                    request.prefill_position,
                    request.prefill_position + chunk_length,
                )
            )
            total_tokens += chunk_length

        for slot_id in range(self.max_num_seqs):
            if total_tokens >= prefill_budget or not self.queue:
                break
            if self.slots[slot_id] is not None:
                continue
            request = self.queue[0]
            prompt_tokens = request.prompt_length
            available = prefill_budget - total_tokens
            if prompt_tokens > prefill_budget:
                if not self.enable_chunked_prefill:
                    raise ValueError("prompt exceeds prefill token budget")
                chunk_length = available
            elif prompt_tokens > available:
                break
            else:
                chunk_length = prompt_tokens
            request = self.queue.popleft()
            request.slot_id = slot_id
            request.status = RequestStatus.PREFILL
            self.slots[slot_id] = request
            request.prefill_position = self.state_manager.restore_prefix(
                slot_id,
                request.inputs.input_ids,
            )
            remaining = request.prompt_length - request.prefill_position
            chunk_length = min(chunk_length, remaining)
            scheduled.append(
                PrefillChunk(
                    request,
                    request.prefill_position,
                    request.prefill_position + chunk_length,
                )
            )
            total_tokens += chunk_length
        return scheduled

    def _run_batched_prefill(
        self,
        work: list[PrefillChunk] | list[Request],
    ) -> None:
        """Execute a batched prefill step for the given requests.

        Args:
            work: Requests or explicit prompt ranges to prefill.

        Raises:
            RuntimeError: If the generation adapter does not return logits.
        """
        chunks = [
            item
            if isinstance(item, PrefillChunk)
            else PrefillChunk(item, item.prefill_position, item.prompt_length)
            for item in work
        ]
        self.prefill_token_history.append(
            sum(chunk.length for chunk in chunks)
        )
        bucket = self._get_bucket_for_len(
            max(chunk.length for chunk in chunks)
        )
        shape = (self.max_num_seqs, bucket)
        input_ids = jnp.zeros(shape, dtype=jnp.int32)
        positions = jnp.zeros(shape, dtype=jnp.int32)
        active_mask = jnp.zeros(shape, dtype=jnp.bool_)
        chunk_lengths = jnp.zeros((self.max_num_seqs,), dtype=jnp.int32)

        for chunk in chunks:
            request = chunk.request
            slot_id = self._slot_id(request)
            length = chunk.length
            tokens = jnp.asarray(
                request.inputs.input_ids[chunk.start:chunk.end],
                dtype=jnp.int32,
            )
            input_ids = input_ids.at[slot_id, :length].set(tokens)
            positions = positions.at[slot_id, :length].set(
                jnp.arange(chunk.start, chunk.end, dtype=jnp.int32)
            )
            active_mask = active_mask.at[slot_id, :length].set(True)
            chunk_lengths = chunk_lengths.at[slot_id].set(length)
            self.state_manager.ensure_length(slot_id, chunk.end)
            self.state_manager.set_sequence_length(slot_id, chunk.end)

        modalities, modality_mask = collate_modalities(
            [
                (self._slot_id(chunk.request), chunk.request.inputs)
                for chunk in chunks
            ],
            self.max_num_seqs,
        )

        output = self.runner.execute(
            Batch(
                input_ids=input_ids,
                positions=positions,
                slot_ids=jnp.arange(self.max_num_seqs, dtype=jnp.int32),
                active_mask=active_mask,
                modalities=modalities,
                modality_mask=modality_mask,
                metadata={
                    "logit_indices": jnp.maximum(chunk_lengths, 1) - 1,
                },
            ),
            phase="prefill",
        )
        completed: list[Request] = []
        for chunk in chunks:
            request = chunk.request
            request.prefill_position = chunk.end
            self.state_manager.publish_prefix(
                self._slot_id(request),
                request.inputs.input_ids,
                request.prefill_position,
            )
            if request.prefill_position == request.prompt_length:
                completed.append(request)
        if not completed:
            return
        if output.logits is None:
            raise RuntimeError("generation adapter did not return logits")
        rows = [self._slot_id(request) for request in completed]
        tokens = self.sampler.sample(
            output.logits[:, 0, :],
            rows,
            [request.sampling_params for request in completed],
            [0] * len(completed),
        )
        for request, token in zip(completed, tokens, strict=True):
            request.generated_tokens.append(token)
            request.position = request.prompt_length
            request.status = RequestStatus.DECODING

    def _run_decode(self, active_slots: list[int]) -> None:
        """Execute a batched decode step for the given active slots.

        Args:
            active_slots (list[int]): The slot indices of active decoding requests.

        Raises:
            RuntimeError: If the generation adapter does not return logits.
        """
        input_ids = jnp.zeros((self.max_num_seqs, 1), dtype=jnp.int32)
        positions = jnp.zeros((self.max_num_seqs, 1), dtype=jnp.int32)
        active_mask = jnp.zeros((self.max_num_seqs, 1), dtype=jnp.bool_)
        requests: list[Request] = []
        for slot_id in active_slots:
            request = self.slots[slot_id]
            if request is None:
                continue
            requests.append(request)
            input_ids = input_ids.at[slot_id, 0].set(
                request.generated_tokens[-1]
            )
            positions = positions.at[slot_id, 0].set(request.position)
            active_mask = active_mask.at[slot_id, 0].set(True)
            self.state_manager.ensure_position(slot_id, request.position)
            self.state_manager.set_sequence_length(
                slot_id,
                request.position + 1,
            )

        modalities, modality_mask = collate_modalities(
            [
                (self._slot_id(request), request.inputs)
                for request in requests
            ],
            self.max_num_seqs,
        )

        output = self.runner.execute(
            Batch(
                input_ids=input_ids,
                positions=positions,
                slot_ids=jnp.arange(self.max_num_seqs, dtype=jnp.int32),
                active_mask=active_mask,
                modalities=modalities,
                modality_mask=modality_mask,
                metadata={
                    "logit_indices": jnp.zeros(
                        (self.max_num_seqs,),
                        dtype=jnp.int32,
                    ),
                },
            ),
            phase="decode",
        )
        if output.logits is None:
            raise RuntimeError("generation adapter did not return logits")
        tokens = self.sampler.sample(
            output.logits[:, 0, :],
            active_slots,
            [request.sampling_params for request in requests],
            [len(request.generated_tokens) for request in requests],
        )
        for request, token in zip(requests, tokens, strict=True):
            request.generated_tokens.append(token)
            request.position += 1

    def _retire_finished(self, requests: list[Request]) -> None:
        """Retire and clean up resources for finished requests.

        Args:
            requests (list[Request]): The requests to check and retire.
        """
        for request in requests:
            if not self._is_finished(request):
                continue
            request.status = RequestStatus.FINISHED
            slot_id = self._slot_id(request)
            self.state_manager.reset_slot(slot_id)
            self.slots[slot_id] = None

    def _is_finished(self, request: Request) -> bool:
        """Check if a request has finished generation.

        Args:
            request (Request): The request to check.

        Returns:
            bool: True if generation is complete, False otherwise.
        """
        if len(request.generated_tokens) >= request.sampling_params.max_tokens:
            return True
        if not request.generated_tokens:
            return False
        stop_ids = request.sampling_params.stop_token_ids
        if stop_ids is None and self.tokenizer is not None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            stop_ids = [eos_id] if eos_id is not None else None
        return stop_ids is not None and request.generated_tokens[-1] in stop_ids

    @staticmethod
    def _slot_id(request: Request) -> int:
        """Retrieve the slot ID for a request.

        Args:
            request (Request): The request to inspect.

        Raises:
            RuntimeError: If the request has no assigned slot.

        Returns:
            int: The slot ID.
        """
        if request.slot_id is None:
            raise RuntimeError("request has no assigned slot")
        return request.slot_id


__all__ = ["Scheduler"]
