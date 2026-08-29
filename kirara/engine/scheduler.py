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

import jax
import jax.numpy as jnp

from kirara.engine.request import Request, RequestStatus
from kirara.engine.runner import Runner
from kirara.engine.sampler import Sampler
from kirara.inputs import NormalizedInput
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
        if policy != "fcfs":
            raise ValueError("only fcfs scheduling is implemented")
        if enable_prefix_caching:
            raise NotImplementedError("prefix caching is not implemented")
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
        self.queue: deque[Request] = deque()
        self.slots: list[Request | None] = [None] * max_num_seqs
        self.next_request_id = 0

    @property
    def cache_manager(self):
        return self.state_manager.paged_cache

    @property
    def prefill_model_invocations(self) -> int:
        return self.runner.prefill_model_invocations

    @property
    def prefill_bucket_history(self) -> list[int]:
        return self.runner.prefill_bucket_history

    @property
    def compiled_prefill_buckets(self) -> set[int]:
        return self.runner.compiled_prefill_buckets

    @property
    def decode_model_invocations(self) -> int:
        return self.runner.decode_model_invocations

    @property
    def decode_compiled(self) -> bool:
        return self.runner.decode_compiled

    @property
    def compiled_executable_count(self) -> int:
        return self.runner.compiled_executable_count

    @property
    def _compiled_prefill_steps(self):
        return self.runner._prefill_steps

    def _get_bucket_for_len(self, length: int) -> int:
        return self.runner.bucket_for_length(length)

    def add_request(
        self,
        inputs: NormalizedInput | list[int],
        sampling_params: SamplingParams | None = None,
    ) -> Request:
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
        prefill_requests = self._admit_prefill()
        prefill_tokens = sum(
            request.prompt_length for request in prefill_requests
        )
        if prefill_requests:
            self._run_batched_prefill(prefill_requests)
            self._retire_finished(prefill_requests)

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
        requests = [
            self.add_request(item, sampling_params)
            for item in inputs
        ]
        while self.step():
            pass
        return [request.generated_tokens for request in requests]

    def _admit_prefill(self) -> list[Request]:
        admitted: list[Request] = []
        total_tokens = 0
        prefill_budget = min(
            self.max_num_prefill_tokens,
            self.max_num_batched_tokens,
        )
        for slot_id in range(self.max_num_seqs):
            if self.slots[slot_id] is not None or not self.queue:
                continue
            request = self.queue[0]
            prompt_tokens = request.prompt_length
            if (
                admitted
                and total_tokens + prompt_tokens
                > prefill_budget
            ):
                break
            if prompt_tokens > prefill_budget:
                if not self.enable_chunked_prefill:
                    raise ValueError("prompt exceeds prefill token budget")
                raise NotImplementedError(
                    "chunked prefill scheduling is not implemented yet"
                )
            request = self.queue.popleft()
            request.slot_id = slot_id
            request.status = RequestStatus.PREFILL
            self.slots[slot_id] = request
            admitted.append(request)
            total_tokens += prompt_tokens
        return admitted

    def _run_batched_prefill(self, requests: list[Request]) -> None:
        max_prompt_length = max(request.prompt_length for request in requests)
        bucket = self._get_bucket_for_len(max_prompt_length)
        shape = (self.max_num_seqs, bucket)
        input_ids = jnp.zeros(shape, dtype=jnp.int32)
        positions = jnp.zeros(shape, dtype=jnp.int32)
        active_mask = jnp.zeros(shape, dtype=jnp.bool_)
        prompt_lengths = jnp.zeros((self.max_num_seqs,), dtype=jnp.int32)

        for request in requests:
            slot_id = self._slot_id(request)
            length = request.prompt_length
            tokens = jnp.asarray(request.inputs.input_ids, dtype=jnp.int32)
            input_ids = input_ids.at[slot_id, :length].set(tokens)
            positions = positions.at[slot_id, :length].set(
                jnp.arange(length, dtype=jnp.int32)
            )
            active_mask = active_mask.at[slot_id, :length].set(True)
            prompt_lengths = prompt_lengths.at[slot_id].set(length)
            self.state_manager.ensure_length(slot_id, length)
            self.state_manager.set_sequence_length(slot_id, length)

        output = self.runner.execute(
            Batch(
                input_ids=input_ids,
                positions=positions,
                slot_ids=jnp.arange(self.max_num_seqs, dtype=jnp.int32),
                active_mask=active_mask,
                metadata={
                    "logit_indices": jnp.maximum(prompt_lengths, 1) - 1,
                },
            ),
            phase="prefill",
        )
        if output.logits is None:
            raise RuntimeError("generation adapter did not return logits")
        rows = [self._slot_id(request) for request in requests]
        tokens = self.sampler.sample(
            output.logits[:, 0, :],
            rows,
            [request.sampling_params for request in requests],
            [0] * len(requests),
        )
        for request, token in zip(requests, tokens, strict=True):
            request.generated_tokens.append(token)
            request.position = request.prompt_length
            request.status = RequestStatus.DECODING

    def _run_decode(self, active_slots: list[int]) -> None:
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

        output = self.runner.execute(
            Batch(
                input_ids=input_ids,
                positions=positions,
                slot_ids=jnp.arange(self.max_num_seqs, dtype=jnp.int32),
                active_mask=active_mask,
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
        for request in requests:
            if not self._is_finished(request):
                continue
            request.status = RequestStatus.FINISHED
            slot_id = self._slot_id(request)
            self.state_manager.reset_slot(slot_id)
            self.slots[slot_id] = None

    def _is_finished(self, request: Request) -> bool:
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
        if request.slot_id is None:
            raise RuntimeError("request has no assigned slot")
        return request.slot_id


__all__ = ["Scheduler"]
