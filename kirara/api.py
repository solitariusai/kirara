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

"""Public LLM facade for the Kirara inference runtime."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from kirara.attention import AttentionBackend, PagedAttention
from kirara.engine import Engine, Output, Runner, Sampler, Scheduler
from kirara.inputs import GenerationInput
from kirara.integrations import register_builtin_integrations
from kirara.models import Adapter, model_registry
from kirara.sampling import SamplingParams
from kirara.sharding import build_mesh
from kirara.state import PagedCacheManager, StateManager
from kirara.types import (
    CacheDType,
    MeshSpec,
    ModelDType,
    ModelSource,
    SchedulerPolicy,
)


class LLM:
    """Small public facade over Kirara's model-agnostic inference runtime."""

    def __init__(
        self,
        model: ModelSource,
        *,
        model_impl: str = "auto",
        tokenizer: str | None = None,
        revision: str | None = None,
        dtype: ModelDType = "bfloat16",
        quantization: Any | None = None,
        mesh: MeshSpec = None,
        max_model_len: int | None = None,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int | None = None,
        max_num_prefill_tokens: int | None = None,
        cache_dtype: CacheDType = "auto",
        block_size: int = 16,
        scheduler_policy: SchedulerPolicy = "fcfs",
        attention_backend: AttentionBackend | str | None = None,
        enable_prefix_caching: bool = False,
        enable_chunked_prefill: bool = True,
        trust_remote_code: bool = False,
        **model_kwargs: Any,
    ) -> None:
        """Initialize the LLM facade.

        Args:
            model (ModelSource): The model source (e.g., Hugging Face model ID).
            model_impl (str, optional): The model implementation to use. Defaults to "auto".
            tokenizer (str | None, optional): The tokenizer name or path. Defaults to None.
            revision (str | None, optional): The model revision. Defaults to None.
            dtype (ModelDType, optional): The data type for the model weights. Defaults to "bfloat16".
            quantization (Any | None, optional): Quantization config. Defaults to None.
            mesh (MeshSpec, optional): Device mesh specification for sharding. Defaults to None.
            max_model_len (int | None, optional): Maximum model context length. Defaults to None.
            max_num_seqs (int, optional): Maximum number of sequences in a batch. Defaults to 256.
            max_num_batched_tokens (int | None, optional): Maximum number of batched tokens. Defaults to None.
            max_num_prefill_tokens (int | None, optional): Maximum number of prefill tokens. Defaults to None.
            cache_dtype (CacheDType, optional): Data type for the KV cache. Defaults to "auto".
            block_size (int, optional): Block size for paged attention. Defaults to 16.
            scheduler_policy (SchedulerPolicy, optional): The scheduling policy. Defaults to "fcfs".
            attention_backend (AttentionBackend | str | None, optional): Attention backend to use. Defaults to None.
            enable_prefix_caching (bool, optional): Whether to enable prefix caching. Defaults to False.
            enable_chunked_prefill (bool, optional): Whether to enable chunked prefill. Defaults to True.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            **model_kwargs (Any): Additional keyword arguments for the model.

        Raises:
            ValueError: If max_num_seqs is less than 1.
            ValueError: If block_size is less than 1.
            ValueError: If the model adapter exposes no executable capability.
            ValueError: If max_model_len is less than 1.
        """
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        if block_size < 1:
            raise ValueError("block_size must be positive")

        register_builtin_integrations(model_registry)
        integration = model_registry.resolve(model, model_impl)
        self.mesh = build_mesh(mesh)
        loaded = integration.loader.load(
            model,
            dtype=dtype,
            revision=revision,
            tokenizer=tokenizer,
            quantization=quantization,
            mesh=self.mesh,
            trust_remote_code=trust_remote_code,
            model_kwargs=dict(model_kwargs),
        )
        self.model_source = model
        self.model_impl = model_impl
        self.loaded_model = loaded
        self.attention_backend = self._resolve_attention_backend(
            attention_backend,
            block_size,
        )
        self.adapter: Adapter = integration.adapter(
            loaded,
            self.attention_backend,
            {"block_size": block_size},
        )
        if not (
            self.adapter.capabilities.generate
            or self.adapter.capabilities.encode
        ):
            raise ValueError("model adapter exposes no executable capability")

        configured_length = self.adapter.config.max_model_len
        self.max_model_len = max_model_len or configured_length or 2048
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be positive")
        prefill_buckets = self._prefill_buckets(self.max_model_len)
        self.state_manager = self._create_state_manager(
            max_num_seqs=max_num_seqs,
            max_model_len=self.max_model_len,
            cache_dtype=cache_dtype,
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.runner = Runner(
            model=self.adapter,
            state_manager=self.state_manager,
            attention_backend=self.attention_backend,
            mesh=self.mesh,
            prefill_buckets=prefill_buckets,
        )
        self.processor = integration.processor(loaded)
        self.sampler = Sampler()
        self.scheduler = Scheduler(
            runner=self.runner,
            state_manager=self.state_manager,
            sampler=self.sampler,
            max_num_seqs=max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_prefill_tokens=max_num_prefill_tokens,
            tokenizer=loaded.tokenizer,
            policy=scheduler_policy,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.engine = Engine(
            model=self.adapter,
            processor=self.processor,
            runner=self.runner,
            scheduler=self.scheduler,
        )

    @property
    def model(self) -> Any:
        """Get the underlying loaded model.

        Returns:
            Any: The loaded model instance.
        """
        return self.loaded_model.model

    def generate(
        self,
        inputs: GenerationInput,
        sampling_params: SamplingParams | None = None,
    ) -> list[Output]:
        """Generate text from the given inputs.

        Args:
            inputs (GenerationInput): The inputs to generate from.
            sampling_params (SamplingParams | None, optional): Sampling parameters for generation. Defaults to None.

        Returns:
            list[Output]: A list of generation outputs.
        """
        return self.engine.generate(inputs, sampling_params)

    def encode(self, inputs: GenerationInput) -> jax.Array:
        """Encode the inputs into embeddings.

        Args:
            inputs (GenerationInput): The inputs to encode.

        Returns:
            jax.Array: The resulting embeddings.
        """
        return self.engine.encode(inputs)

    def _create_state_manager(
        self,
        *,
        max_num_seqs: int,
        max_model_len: int,
        cache_dtype: CacheDType,
        block_size: int,
        enable_prefix_caching: bool,
    ) -> StateManager:
        """Create paged and per-slot persistent model state.

        Args:
            max_num_seqs (int): Maximum number of sequences.
            max_model_len (int): Maximum model length.
            cache_dtype (CacheDType): Data type for the cache.
            block_size (int): Block size for paged attention.
            enable_prefix_caching (bool): Retain reusable paged prompt prefixes.

        Raises:
            ValueError: If the adapter requires layer, KV-head, and head sizes but they are invalid.
            ValueError: If the cache dtype is unsupported.

        Returns:
            StateManager: The created state manager.
        """
        state_spec = self.adapter.state_spec
        if state_spec is None:
            if enable_prefix_caching:
                raise ValueError("prefix caching requires paged KV state")
            return StateManager()
        config = self.adapter.config
        dtype = config.dtype if cache_dtype == "auto" else cache_dtype
        if dtype is None and cache_dtype == "auto":
            dtype = jnp.float32
        if isinstance(dtype, str):
            dtype = getattr(jnp, dtype, None)
        if dtype is None:
            raise ValueError(f"unsupported cache dtype: {cache_dtype}")
        cache = None
        if "kv" in state_spec.kinds:
            if min(
                config.num_layers,
                config.num_kv_heads,
                config.head_dim,
            ) < 1:
                raise ValueError(
                    "paged-state adapters require layer, KV-head, and head sizes"
                )
            cache = PagedCacheManager(
                max_batch_size=max_num_seqs,
                max_seq_len=max_model_len,
                num_layers=config.num_layers,
                num_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                dtype=dtype,
                block_size=block_size,
            )

        slot_state: dict[str, jax.Array] = {}
        for kind in state_spec.kinds:
            if kind == "kv":
                continue
            try:
                shape = state_spec.shapes[kind]
            except KeyError as error:
                raise ValueError(
                    f"persistent state {kind!r} requires a static shape"
                ) from error
            state_dtype = state_spec.dtypes.get(kind, dtype)
            slot_state[kind] = jnp.zeros(
                (max_num_seqs, *shape),
                dtype=state_dtype,
            )
        if enable_prefix_caching and slot_state:
            raise ValueError(
                "prefix caching currently requires pure paged KV state; "
                "recurrent prefix snapshots need model-level scan outputs"
            )
        return StateManager(
            paged_cache=cache,
            slot_state=slot_state,
            enable_prefix_caching=enable_prefix_caching,
        )

    @staticmethod
    def _resolve_attention_backend(
        backend: AttentionBackend | str | None,
        block_size: int,
    ) -> AttentionBackend:
        """Resolve the attention backend.

        Args:
            backend (AttentionBackend | str | None): The attention backend to use.
            block_size (int): The block size.

        Raises:
            ValueError: If the backend string is unknown.
            TypeError: If the attention backend is not callable.

        Returns:
            AttentionBackend: The resolved attention backend.
        """
        if backend is None or backend == "paged":
            return PagedAttention(block_size)
        if isinstance(backend, str):
            raise ValueError(f"unknown attention backend: {backend}")
        if not callable(backend):
            raise TypeError("attention_backend must be callable")
        return backend

    @staticmethod
    def _prefill_buckets(max_model_len: int) -> tuple[int, ...]:
        """Calculate the prefill buckets based on max model length.

        Args:
            max_model_len (int): Maximum model length.

        Returns:
            tuple[int, ...]: A sorted tuple of bucket sizes.
        """
        buckets: list[int] = []
        bucket = 16
        while bucket < max_model_len:
            buckets.append(bucket)
            bucket *= 2
        buckets.append(max_model_len)
        return tuple(sorted(set(buckets)))


__all__ = [
    "LLM",
    "SamplingParams",
]
