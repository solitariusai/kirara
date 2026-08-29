from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from taktiny.cosettes.transformers.ordinario import StackLayer

from kirara.cache import write_paged_kv
from kirara.kernels import paged_attention


class TakTinyModelRunner:
    """Execute a TakTiny-backed model with Kirara's paged KV kernels."""

    def __init__(self, model: Any, block_size: int) -> None:
        self.model = model
        self.block_size = block_size
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
            attended = paged_attention(
                query,
                layer_key_pool,
                layer_value_pool,
                block_table,
                sequence_lengths,
                position_ids,
                token_active,
                block_size=self.block_size,
                scale=attention.scaling,
                softcap=attention.softcap,
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


__all__ = ["TakTinyModelRunner"]
