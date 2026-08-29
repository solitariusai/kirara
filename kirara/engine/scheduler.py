from collections import deque
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from kirara.cache import PagedCacheManager
from kirara.api.schemas import SamplingParams
from kirara.models.taktiny import TakTinyModelRunner


@dataclass
class Request:
    request_id: int
    prompt_ids: list[int]
    sampling_params: SamplingParams
    generated_tokens: list[int]
    status: str = "PENDING"
    slot_id: int = -1
    pos: int = 0

    @property
    def max_new_tokens(self) -> int:
        return self.sampling_params.max_new_tokens


class Scheduler:
    """Continuously batch prefill and decode over immutable cache slots."""

    def __init__(
        self,
        model: Any,
        max_batch_size: int,
        max_seq_len: int,
        tokenizer: Any = None,
        prefill_token_budget: int | None = None,
        prefill_buckets: list[int] | None = None,
        block_size: int = 16,
        num_kv_blocks: int | None = None,
    ) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        # The prefill budget is the maximum prompt-token sum admitted at once.
        if prefill_token_budget is None:
            prefill_token_budget = max_seq_len
        self.prefill_token_budget = int(prefill_token_budget)
        if self.prefill_token_budget < 1:
            raise ValueError("prefill_token_budget must be positive")
        # Static powers-of-two buckets bound the number of JAX executables.
        if prefill_buckets is None:
            buckets = []
            b = 16
            while b < max_seq_len:
                buckets.append(b)
                b *= 2
            buckets.append(max_seq_len)
            self.prefill_buckets = sorted(set(buckets))
        else:
            self.prefill_buckets = sorted(set(prefill_buckets))
        if not self.prefill_buckets or any(
            bucket < 1 or bucket > max_seq_len
            for bucket in self.prefill_buckets
        ):
            raise ValueError(
                "prefill_buckets must contain values between 1 and max_seq_len"
            )

        self.queue: deque[Request] = deque()
        self.slots: list[Request | None] = [None] * max_batch_size
        self.next_request_id = 0

        # Initialize the one authoritative paged KV pool.
        config = getattr(
            self.model,
            "config",
            getattr(self.model, "_default_config", None),
        )
        if config is None:
            raise TypeError(
                "native Kirara models must expose config or _default_config"
            )
        num_layers = getattr(config, "num_hidden_layers", 32)
        num_heads = (
            getattr(config, "num_key_value_heads", None)
            or getattr(config, "num_attention_heads", 32)
        )
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(config, "hidden_size", 4096)
            attention_heads = getattr(config, "num_attention_heads", 32)
            head_dim = hidden_size // attention_heads
        dtype = getattr(
            config,
            "dtype",
            getattr(config, "torch_dtype", jnp.float32),
        )
        if isinstance(dtype, str):
            dtype = getattr(jnp, dtype, jnp.float32)
        self.cache_manager = PagedCacheManager(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            block_size=block_size,
            num_blocks=num_kv_blocks,
        )
        self.model_runner = TakTinyModelRunner(
            self.model,
            block_size=block_size,
        )

        # Useful both for observability and for verifying that the number of
        # prefill executables stays bounded by ``prefill_buckets``.
        self.prefill_model_invocations = 0
        self.prefill_bucket_history: list[int] = []
        self.compiled_prefill_buckets: set[int] = set()
        self.decode_model_invocations = 0
        self.decode_compiled = False

        # JIT-compiled decode step function
        self._compiled_decode_step = None
        self._init_compiled_steps()

    def _get_bucket_for_len(self, seq_len: int) -> int:
        """Return the smallest configured static bucket containing seq_len."""
        if seq_len < 1:
            raise ValueError("prompts must contain at least one token")
        for b in self.prefill_buckets:
            if seq_len <= b:
                return b
        raise ValueError(
            f"prompt length {seq_len} exceeds the largest prefill bucket "
            f"({self.prefill_buckets[-1]})"
        )

    @property
    def compiled_executable_count(self) -> int:
        """Number of scheduler JIT executables materialized so far."""
        return len(self.compiled_prefill_buckets) + int(self.decode_compiled)

    def _init_compiled_steps(self) -> None:
        @jax.jit
        def decode_step(
            input_ids,
            position_ids,
            token_active,
            block_table,
            sequence_lengths,
            key_pool,
            value_pool,
        ):
            logits, updated_cache = self.model_runner(
                input_ids,
                position_ids,
                token_active,
                block_table,
                sequence_lengths,
                key_pool,
                value_pool,
                jnp.zeros((self.max_batch_size,), dtype=jnp.int32),
            )
            return logits, updated_cache

        self._compiled_decode_step = decode_step
        self._compiled_prefill_steps = {}

    def add_request(
        self,
        input_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ) -> Request:
        self._get_bucket_for_len(len(input_ids))
        parameters = sampling_params or SamplingParams()
        req = Request(
            request_id=self.next_request_id,
            prompt_ids=list(input_ids),
            sampling_params=parameters,
            generated_tokens=[],
        )
        self.queue.append(req)
        self.next_request_id += 1
        return req

    def _is_eos(self, req: Request) -> bool:
        if not req.generated_tokens:
            return False
        last_tok = req.generated_tokens[-1]

        eos_ids = req.sampling_params.eos_token_ids
        if (
            eos_ids is None
            and self.tokenizer is not None
            and hasattr(self.tokenizer, "eos_token_id")
        ):
            eos_ids = self.tokenizer.eos_token_id

        if eos_ids is None:
            return False

        if isinstance(eos_ids, int):
            return last_tok == eos_ids
        if isinstance(eos_ids, (list, tuple, set)):
            return last_tok in eos_ids
        return False

    def step(self) -> bool:
        """Perform one scheduling step.

        1. Allocates available slots to queued requests and runs batched prefill
            (single model invocation per batch, respecting token budget).
        2. Decodes all active requests in one fixed-shape model invocation.
        3. Retires finished requests and clears their slots.
        """
        # 1. Fill empty slots with pending requests and run batched prefill
        # Respect prefill_token_budget: sum of prompt tokens in this batch
        prefill_reqs: list[Request] = []
        total_prefill_tokens = 0
        for s in range(self.max_batch_size):
            if self.slots[s] is None and self.queue:
                peek = self.queue[0]
                peek_len = len(peek.prompt_ids)
                # If adding this would exceed budget and we already have some, stop
                if (
                    prefill_reqs
                    and total_prefill_tokens + peek_len
                    > self.prefill_token_budget
                ):
                    break
                # Also handle single huge prompt that exceeds budget: still admit one
                req = self.queue.popleft()
                req.slot_id = s
                req.status = "PREFILL"
                self.slots[s] = req
                prefill_reqs.append(req)
                total_prefill_tokens += peek_len

        if prefill_reqs:
            self._run_batched_prefill(prefill_reqs)

            # Check if prompt prefill generated all requested tokens or eos
            for req in prefill_reqs:
                if len(req.generated_tokens) >= req.max_new_tokens or self._is_eos(req):
                    req.status = "FINISHED"
                    if self.cache_manager is not None:
                        self.cache_manager.reset_slot(req.slot_id)
                    self.slots[req.slot_id] = None

        # 2. Decode for all active slots
        active_slots = [
            slot_id
            for slot_id, request in enumerate(self.slots)
            if request is not None and request.status == "DECODE"
        ]
        if active_slots:
            self._run_decode(active_slots)

            # Check for finished requests
            for s in active_slots:
                req = self.slots[s]
                if req is not None and (
                    len(req.generated_tokens) >= req.max_new_tokens
                    or self._is_eos(req)
                ):
                    req.status = "FINISHED"
                    if self.cache_manager is not None:
                        self.cache_manager.reset_slot(s)
                    self.slots[s] = None

        # Check if there are any remaining requests
        has_active = any(s is not None for s in self.slots)
        has_queued = bool(self.queue)
        return has_active or has_queued

    def _get_compiled_prefill_step(self, bucket_size: int):
        """Return the one compiled prefill executable for ``bucket_size``.

        Bucket size is represented only by array shapes.  Prompt lengths stay
        dynamic array values, so prompts within a bucket share an executable.
        """
        if bucket_size not in self._compiled_prefill_steps:
            @jax.jit
            def prefill_step(
                input_ids: jax.Array,
                position_ids: jax.Array,
                token_active: jax.Array,
                prompt_lengths: jax.Array,
                block_table: jax.Array,
                key_pool: jax.Array,
                value_pool: jax.Array,
            ):
                return self.model_runner(
                    input_ids,
                    position_ids,
                    token_active,
                    block_table,
                    prompt_lengths,
                    key_pool,
                    value_pool,
                    jnp.maximum(prompt_lengths, 1) - 1,
                )

            self._compiled_prefill_steps[bucket_size] = prefill_step
        return self._compiled_prefill_steps[bucket_size]

    def _run_batched_prefill(self, reqs: list[Request]) -> None:
        if self.cache_manager is None:
            for req in reqs:
                req.generated_tokens.append(1)
                req.pos = len(req.prompt_ids) + 1
                req.status = "DECODE"
            return

        # Determine bucket for the batch (finite set)
        max_prompt_len = max(len(req.prompt_ids) for req in reqs)
        bucket_size = self._get_bucket_for_len(max_prompt_len)

        # All arrays have one of a finite set of shapes.  Rows are indexed by
        # immutable cache slot IDs rather than by admission order.
        shape = (self.max_batch_size, bucket_size)
        input_batch = jnp.zeros(shape, dtype=jnp.int32)
        position_batch = jnp.zeros(shape, dtype=jnp.int32)
        token_active = jnp.zeros(shape, dtype=jnp.bool_)
        prompt_lengths = jnp.zeros((self.max_batch_size,), dtype=jnp.int32)

        for req in reqs:
            s = req.slot_id
            seq_len = len(req.prompt_ids)
            prompt_ids = jnp.asarray(req.prompt_ids, dtype=jnp.int32)
            input_batch = input_batch.at[s, :seq_len].set(prompt_ids)
            pos = jnp.arange(seq_len, dtype=jnp.int32)
            position_batch = position_batch.at[s, :seq_len].set(pos)
            token_active = token_active.at[s, :seq_len].set(True)
            prompt_lengths = prompt_lengths.at[s].set(seq_len)
            self.cache_manager.ensure_length(s, seq_len)
            self.cache_manager.set_sequence_length(s, seq_len)

        # Inactive rows use index zero only for the irrelevant logit gather;
        # their cache write mask remains empty.  Admitted rows gather the real
        # last prompt token, never the bucket's padded tail.
        prefill_step = self._get_compiled_prefill_step(bucket_size)
        logits, updated_cache = prefill_step(
            input_batch,
            position_batch,
            token_active,
            prompt_lengths,
            self.cache_manager.block_table,
            self.cache_manager.cache[0],
            self.cache_manager.cache[1],
        )
        self.cache_manager.cache = updated_cache
        self.prefill_model_invocations += 1
        self.prefill_bucket_history.append(bucket_size)
        self.compiled_prefill_buckets.add(bucket_size)

        first_tokens = jax.device_get(jnp.argmax(logits[:, 0, :], axis=-1))
        for req in reqs:
            req.generated_tokens.append(int(first_tokens[req.slot_id]))
            req.pos = len(req.prompt_ids)
            req.status = "DECODE"

    def _run_decode(self, active_slots: list[int]) -> None:
        if self.cache_manager is None:
            for s in active_slots:
                req = self.slots[s]
                req.generated_tokens.append(1)
                req.pos += 1
            return

        # The batch shape never depends on the number or identity of active
        # slots, so occupancy changes reuse the same compiled executable.
        input_batch = jnp.zeros((self.max_batch_size, 1), dtype=jnp.int32)
        pos_batch = jnp.zeros((self.max_batch_size, 1), dtype=jnp.int32)
        token_active = jnp.zeros((self.max_batch_size, 1), dtype=jnp.bool_)
        for s in active_slots:
            req = self.slots[s]
            input_batch = input_batch.at[s, 0].set(req.generated_tokens[-1])
            pos_batch = pos_batch.at[s, 0].set(req.pos)
            token_active = token_active.at[s, 0].set(True)
            self.cache_manager.ensure_position(s, req.pos)
            self.cache_manager.set_sequence_length(s, req.pos + 1)

        logits, updated_cache = self._compiled_decode_step(
            input_batch,
            pos_batch,
            token_active,
            self.cache_manager.block_table,
            self.cache_manager.sequence_lengths,
            self.cache_manager.cache[0],
            self.cache_manager.cache[1],
        )
        self.cache_manager.cache = updated_cache
        self.decode_model_invocations += 1
        self.decode_compiled = True

        next_tokens = jax.device_get(jnp.argmax(logits[:, 0, :], axis=-1))
        for s in active_slots:
            req = self.slots[s]
            req.generated_tokens.append(int(next_tokens[s]))
            req.pos += 1

    def generate(
        self,
        input_ids_list: list[list[int]],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[int]]:
        """Generate token IDs for a batch of pretokenized prompts."""
        requests = [
            self.add_request(input_ids, sampling_params)
            for input_ids in input_ids_list
        ]
        while self.step():
            pass
        return [request.generated_tokens for request in requests]
