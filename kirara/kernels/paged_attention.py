from __future__ import annotations

import jax
import jax.numpy as jnp


def paged_attention(
    query: jax.Array,
    key_pool: jax.Array,
    value_pool: jax.Array,
    block_table: jax.Array,
    sequence_lengths: jax.Array,
    query_positions: jax.Array,
    query_active: jax.Array,
    *,
    block_size: int,
    scale: float | None = None,
    softcap: float | None = None,
) -> jax.Array:
    """Causal MHA/MQA/GQA over page tiles with a global score softmax.

    Only one physical block is gathered per loop iteration.  A contiguous
    per-request KV sequence is never constructed.
    """
    batch_size, query_length, query_heads, head_dim = query.shape
    kv_heads = key_pool.shape[2]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    groups = query_heads // kv_heads
    if value_pool.shape != key_pool.shape:
        raise ValueError("key and value pools must have identical shapes")
    if key_pool.shape[1] != block_size:
        raise ValueError("pool block dimension does not match block_size")
    if scale is None:
        scale = head_dim**-0.5

    q = query.reshape(
        batch_size,
        query_length,
        kv_heads,
        groups,
        head_dim,
    )
    scores_by_block = jnp.zeros(
        (
            batch_size,
            kv_heads,
            groups,
            query_length,
            block_table.shape[1],
            block_size,
        ),
        dtype=jnp.float32,
    )
    block_offsets = jnp.arange(block_size, dtype=jnp.int32)
    mask_value = jnp.asarray(-1.0e30, dtype=jnp.float32)

    def score_block(logical_block, all_scores):
        physical_ids = block_table[:, logical_block]
        safe_ids = jnp.maximum(physical_ids, 0)
        block_keys = key_pool[safe_ids]
        scores = jnp.einsum(
            "bqhgd,bkhd->bhgqk",
            q,
            block_keys,
            preferred_element_type=jnp.float32,
        ) * scale
        if softcap is not None:
            scores = softcap * jnp.tanh(scores / softcap)

        key_positions = logical_block * block_size + block_offsets
        valid_keys = key_positions[None, :] < sequence_lengths[:, None]
        causal = (
            key_positions[None, None, :]
            <= query_positions[:, :, None]
        )
        valid = (
            valid_keys[:, None, :]
            & causal
            & query_active[:, :, None]
            & (physical_ids >= 0)[:, None, None]
        )
        scores = jnp.where(valid[:, None, None], scores, mask_value)
        return all_scores.at[..., logical_block, :].set(scores)

    scores_by_block = jax.lax.fori_loop(
        0,
        block_table.shape[1],
        score_block,
        scores_by_block,
    )
    probabilities = jax.nn.softmax(
        scores_by_block.reshape(
            batch_size,
            kv_heads,
            groups,
            query_length,
            block_table.shape[1] * block_size,
        ),
        axis=-1,
    ).astype(key_pool.dtype)
    probabilities = probabilities.reshape(scores_by_block.shape)
    accumulator = jnp.zeros(
        (batch_size, kv_heads, groups, query_length, head_dim),
        dtype=query.dtype,
    )

    def apply_value_block(logical_block, current_accumulator):
        physical_ids = block_table[:, logical_block]
        safe_ids = jnp.maximum(physical_ids, 0)
        block_values = value_pool[safe_ids]
        block_output = jnp.einsum(
            "bhgqk,bkhd->bhgqd",
            probabilities[..., logical_block, :],
            block_values,
        )
        return current_accumulator + block_output

    accumulator = jax.lax.fori_loop(
        0,
        block_table.shape[1],
        apply_value_block,
        accumulator,
    )
    output = accumulator.transpose(0, 3, 1, 2, 4).reshape(
        batch_size,
        query_length,
        query_heads,
        head_dim,
    )
    output = jnp.where(
        query_active[:, :, None, None],
        output,
        0,
    )
    return output.astype(query.dtype)
