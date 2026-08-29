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

"""TakTiny model execution through Kirara runtime contracts."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from taktiny.cosettes.transformers.ordinario import StackLayer

from kirara.attention import AttentionBackend
from kirara.models.base import (
    Batch,
    Capabilities,
    Config,
    Output,
    StateSpec,
)
from kirara.state import PagedKVState, State, write_paged_kv


class TakTinyModelRunner:
    """Execute a TakTiny-backed model with Kirara's paged KV kernels."""

    def __init__(
        self,
        model: Any,
        block_size: int,
        attention_backend: AttentionBackend,
    ) -> None:
        """Initializes the runner for a TakTiny model.

        Args:
            model (Any): The TakTiny model instance.
            block_size (int): The paged KV block size.
            attention_backend (AttentionBackend): The attention backend implementation.

        Raises:
            TypeError: If the model is not a valid TakTiny causal LM.
        """
        self.model = model
        self.block_size = block_size
        self.attention_backend = attention_backend
        self._custom_forward = getattr(model, "paged_forward", None)
        if self._custom_forward is None:
            base = getattr(model, "model", None)
            required = ("token_embedding", "layers", "norm")
            if base is None or any(not hasattr(base, name) for name in required):
                raise TypeError(
                    "native models must be TakTiny causal LMs or provide "
                    "paged_forward"
                )

    def __call__(
        self,
        input_ids: jax.Array,
        position_ids: jax.Array,
        token_active: jax.Array,
        block_table: jax.Array,
        sequence_lengths: jax.Array,
        key_pool: jax.Array,
        value_pool: jax.Array,
        logit_indices: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """Executes the model forward pass using the paged KV cache.

        Args:
            input_ids (jax.Array): Input token IDs.
            position_ids (jax.Array): Token position IDs.
            token_active (jax.Array): Boolean mask of active tokens.
            block_table (jax.Array): Paged KV block table.
            sequence_lengths (jax.Array): Length of each sequence.
            key_pool (jax.Array): Paged key cache pool.
            value_pool (jax.Array): Paged value cache pool.
            logit_indices (jax.Array): Indices to extract logits for.

        Raises:
            NotImplementedError: If sliding-window attention is configured.

        Returns:
            tuple[jax.Array, tuple[jax.Array, jax.Array]]: The extracted logits and updated (key_pool, value_pool).
        """
        if self._custom_forward is not None:
            return self._custom_forward(
                input_ids=input_ids,
                position_ids=position_ids,
                token_active=token_active,
                block_table=block_table,
                sequence_lengths=sequence_lengths,
                key_pool=key_pool,
                value_pool=value_pool,
                logit_indices=logit_indices,
                block_size=self.block_size,
            )

        transformer = self.model.model
        x = transformer.token_embedding(input_ids, out_sharding=None)
        position_embedding = transformer._position_embeddings(x, position_ids)

        def forward(layer, hidden_states, layer_cache, layer_idx):
            """Forward pass for a single transformer layer.

            Args:
                layer (Any): The transformer layer module.
                hidden_states (jax.Array): The input hidden states.
                layer_cache (tuple[jax.Array, jax.Array]): Layer specific (key, value) pools.
                layer_idx (int): The index of the layer.

            Raises:
                NotImplementedError: If sliding window attention is configured.

            Returns:
                tuple[jax.Array, tuple[jax.Array, jax.Array]]: Updated hidden states and layer cache.
            """
            del layer_idx
            layer_key_pool, layer_value_pool = layer_cache
            residual = hidden_states
            normalized = layer.norm1(hidden_states, out_sharding=None)
            attention = layer.attention
            if attention.window_size is not None:
                raise NotImplementedError(
                    "sliding-window attention is out of scope"
                )

            query = attention.q_proj(normalized)
            key = attention.k_proj(normalized)
            value = attention.v_proj(normalized)
            if attention.q_norm is not None:
                query = attention.q_norm(query)
            if attention.k_norm is not None:
                key = attention.k_norm(key)
            if attention.apply_position_fn is not None:
                query, key = attention.apply_position_fn(
                    query,
                    key,
                    position_embedding,
                )

            layer_key_pool = write_paged_kv(
                layer_key_pool,
                key,
                block_table,
                position_ids,
                token_active,
                self.block_size,
            )
            layer_value_pool = write_paged_kv(
                layer_value_pool,
                value,
                block_table,
                position_ids,
                token_active,
                self.block_size,
            )
            layer_state = State(
                kv=PagedKVState(
                    key_pool=layer_key_pool,
                    value_pool=layer_value_pool,
                    block_table=block_table,
                    sequence_lengths=sequence_lengths,
                )
            )
            attended = self.attention_backend(
                query,
                layer_state,
                {
                    "query_positions": position_ids,
                    "query_active": token_active,
                    "scale": attention.scaling,
                    "softcap": attention.softcap,
                },
            )
            hidden_states = attention.o_proj(attended, out_sharding=None)
            hidden_states = hidden_states + residual
            hidden_states = layer.ffn(
                layer.norm2(hidden_states, out_sharding=None),
                out_sharding=None,
            ) + hidden_states
            return hidden_states, (layer_key_pool, layer_value_pool)

        x, updated_cache = StackLayer.call_stack(
            transformer.layers,
            forward,
            x,
            per_layer=(key_pool, value_pool),
            with_layer_index=True,
        )
        x = transformer.norm(x, out_sharding=None)
        indices = jnp.asarray(logit_indices, dtype=jnp.int32)
        x = jnp.take_along_axis(x, indices[:, None, None], axis=1)
        logits = self.model.compute_logits(
            x,
            self.model._lm_weight(),
            out_sharding=None,
        )
        logits = self.model._process_logits(logits)
        return logits, updated_cache


class TakTinyAdapter:
    """Translate TakTiny models into Kirara's unified adapter contract."""

    def __init__(
        self,
        model: Any,
        attention_backend: AttentionBackend,
        block_size: int,
    ) -> None:
        """Initializes the adapter for a TakTiny model.

        Args:
            model (Any): The TakTiny model instance.
            attention_backend (AttentionBackend): The attention backend to use.
            block_size (int): The paged KV block size.

        Raises:
            TypeError: If the model lacks a configuration object.
        """
        source_config = getattr(
            model,
            "config",
            getattr(model, "_default_config", None),
        )
        if source_config is None:
            raise TypeError("TakTiny models must expose model configuration")
        attention_heads = getattr(source_config, "num_attention_heads", 0)
        kv_heads = (
            getattr(source_config, "num_key_value_heads", None)
            or attention_heads
        )
        head_dim = getattr(source_config, "head_dim", None)
        if head_dim is None and attention_heads:
            head_dim = getattr(source_config, "hidden_size", 0) // attention_heads
        self._config = Config(
            num_layers=getattr(source_config, "num_hidden_layers", 0),
            num_attention_heads=attention_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim or 0,
            max_model_len=getattr(
                source_config,
                "max_position_embeddings",
                None,
            ),
            dtype=getattr(
                source_config,
                "dtype",
                getattr(source_config, "torch_dtype", jnp.float32),
            ),
        )
        self._capabilities = Capabilities(generate=True, stateful=True)
        self._state_spec = StateSpec(kinds=("kv",))
        self._runner = TakTinyModelRunner(
            model,
            block_size,
            attention_backend,
        )

    @property
    def config(self) -> Config:
        """Model configuration.

        Returns:
            Config: The configuration object.
        """
        return self._config

    @property
    def capabilities(self) -> Capabilities:
        """Adapter capabilities.

        Returns:
            Capabilities: The capabilities object.
        """
        return self._capabilities

    @property
    def state_spec(self) -> StateSpec:
        """State specification.

        Returns:
            StateSpec: The state spec object.
        """
        return self._state_spec

    def __call__(
        self,
        batch: Batch,
        state: State | None = None,
    ) -> Output:
        """Executes a single generation step.

        Args:
            batch (Batch): The batched input data.
            state (State | None, optional): The current model state. Defaults to None.

        Raises:
            ValueError: If token or position arrays are missing.
            ValueError: If an active mask is missing.
            ValueError: If paged KV state is missing.
            ValueError: If logit_indices are missing from batch metadata.

        Returns:
            Output: The generation output and updated state.
        """
        if batch.input_ids is None or batch.positions is None:
            raise ValueError("TakTiny generation requires token and position arrays")
        if batch.active_mask is None:
            raise ValueError("TakTiny generation requires an active mask")
        if state is None or "kv" not in state:
            raise ValueError("TakTiny generation requires paged KV state")
        if batch.metadata is None or "logit_indices" not in batch.metadata:
            raise ValueError("batch metadata requires logit_indices")

        kv = state.kv
        logits, (key_pool, value_pool) = self._runner(
            batch.input_ids,
            batch.positions,
            batch.active_mask,
            kv.block_table,
            kv.sequence_lengths,
            kv.key_pool,
            kv.value_pool,
            batch.metadata["logit_indices"],
        )
        values = dict(state.items())
        values["kv"] = PagedKVState(
            key_pool=key_pool,
            value_pool=value_pool,
            block_table=kv.block_table,
            sequence_lengths=kv.sequence_lengths,
        )
        return Output(logits=logits, state=State(**values))


__all__ = ["TakTinyAdapter", "TakTinyModelRunner"]
