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
        """_summary_

        Args:
            model (ModelSource): _description_
            model_impl (str, optional): _description_. Defaults to "auto".
            tokenizer (str | None, optional): _description_. Defaults to None.
            revision (str | None, optional): _description_. Defaults to None.
            dtype (ModelDType, optional): _description_. Defaults to "bfloat16".
            quantization (Any | None, optional): _description_. Defaults to None.
            mesh (MeshSpec, optional): _description_. Defaults to None.
            max_model_len (int | None, optional): _description_. Defaults to None.
            max_num_seqs (int, optional): _description_. Defaults to 256.
            max_num_batched_tokens (int | None, optional): _description_. Defaults to None.
            max_num_prefill_tokens (int | None, optional): _description_. Defaults to None.
            cache_dtype (CacheDType, optional): _description_. Defaults to "auto".
            block_size (int, optional): _description_. Defaults to 16.
            scheduler_policy (SchedulerPolicy, optional): _description_. Defaults to "fcfs".
            attention_backend (AttentionBackend | str | None, optional): _description_. Defaults to None.
            enable_prefix_caching (bool, optional): _description_. Defaults to False.
            enable_chunked_prefill (bool, optional): _description_. Defaults to True.
            trust_remote_code (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
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
        """_summary_

        Returns:
            Any: _description_
        """
        return self.loaded_model.model

    def generate(
        self,
        inputs: GenerationInput,
        sampling_params: SamplingParams | None = None,
    ) -> list[Output]:
        """_summary_

        Args:
            inputs (GenerationInput): _description_
            sampling_params (SamplingParams | None, optional): _description_. Defaults to None.

        Returns:
            list[Output]: _description_
        """
        return self.engine.generate(inputs, sampling_params)

    def encode(self, inputs: GenerationInput) -> jax.Array:
        """_summary_

        Args:
            inputs (GenerationInput): _description_

        Returns:
            jax.Array: _description_
        """
        return self.engine.encode(inputs)

    def _create_state_manager(
        self,
        *,
        max_num_seqs: int,
        max_model_len: int,
        cache_dtype: CacheDType,
        block_size: int,
    ) -> StateManager:
        """_summary_

        Args:
            max_num_seqs (int): _description_
            max_model_len (int): _description_
            cache_dtype (CacheDType): _description_
            block_size (int): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            StateManager: _description_
        """
        state_spec = self.adapter.state_spec
        if state_spec is None or "kv" not in state_spec.kinds:
            return StateManager()
        config = self.adapter.config
        if min(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
        ) < 1:
            raise ValueError(
                "paged-state adapters require layer, KV-head, and head sizes"
            )
        dtype = config.dtype if cache_dtype == "auto" else cache_dtype
        if isinstance(dtype, str):
            dtype = getattr(jnp, dtype, None)
        if dtype is None:
            raise ValueError(f"unsupported cache dtype: {cache_dtype}")
        cache = PagedCacheManager(
            max_batch_size=max_num_seqs,
            max_seq_len=max_model_len,
            num_layers=config.num_layers,
            num_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dtype=dtype,
            block_size=block_size,
        )
        return StateManager(paged_cache=cache)

    @staticmethod
    def _resolve_attention_backend(
        backend: AttentionBackend | str | None,
        block_size: int,
    ) -> AttentionBackend:
        """_summary_

        Args:
            backend (AttentionBackend | str | None): _description_
            block_size (int): _description_

        Raises:
            ValueError: _description_
            TypeError: _description_

        Returns:
            AttentionBackend: _description_
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
        """_summary_

        Args:
            max_model_len (int): _description_

        Returns:
            tuple[int, ...]: _description_
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
