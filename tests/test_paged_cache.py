import unittest

import jax
import jax.numpy as jnp

from kirara.cache import PagedCacheManager, write_paged_kv
from kirara.kernels import paged_attention


class PagedCacheAllocatorTest(unittest.TestCase):
    def make_cache(self, num_blocks=8):
        return PagedCacheManager(
            max_batch_size=3,
            max_seq_len=12,
            num_layers=2,
            num_heads=2,
            head_dim=3,
            block_size=4,
            num_blocks=num_blocks,
        )

    def test_block_boundaries_and_noncontiguous_physical_blocks(self):
        cache = self.make_cache()
        first = cache.ensure_position(0, 0)
        self.assertEqual(cache.ensure_position(0, 3), first)
        intervening = cache.ensure_position(1, 0)
        second = cache.ensure_position(0, 4)
        self.assertEqual(cache.ensure_position(0, 7), second)
        third = cache.ensure_position(0, 8)
        self.assertEqual(cache.ensure_position(0, 11), third)

        self.assertEqual(cache.slot_blocks(0), (first, second, third))
        self.assertEqual((first, intervening, second, third), (0, 1, 2, 3))
        self.assertNotEqual(second, first + 1)
        self.assertEqual(cache.allocated_block_count[0], 3)

    def test_freed_block_is_zeroed_and_reused(self):
        cache = self.make_cache(num_blocks=3)
        old_block = cache.ensure_position(0, 0)
        cache.cache = (
            cache.cache[0].at[:, old_block].set(7),
            cache.cache[1].at[:, old_block].set(9),
        )
        cache.set_sequence_length(0, 4)
        cache.reset_slot(0)

        reused_block = cache.ensure_position(2, 0)
        self.assertEqual(reused_block, old_block)
        self.assertTrue(jnp.all(cache.cache[0][:, reused_block] == 0))
        self.assertTrue(jnp.all(cache.cache[1][:, reused_block] == 0))
        self.assertEqual(cache.block_table[0, 0], -1)
        self.assertEqual(cache.sequence_lengths[0], 0)


class PagedAttentionTest(unittest.TestCase):
    def test_dense_equivalence_for_mha_gqa_and_mqa(self):
        block_size = 4
        max_length = 12
        lengths_to_test = (1, 3, 4, 5, 8, 10)

        for kv_heads in (1, 2, 4):
            query_heads = 4
            with self.subTest(kv_heads=kv_heads):
                key = jax.random.key(100 + kv_heads)
                q_key, k_key, v_key = jax.random.split(key, 3)
                query = jax.random.normal(
                    q_key,
                    (len(lengths_to_test), max_length, query_heads, 8),
                )
                dense_key = jax.random.normal(
                    k_key,
                    (len(lengths_to_test), max_length, kv_heads, 8),
                )
                dense_value = jax.random.normal(
                    v_key,
                    (len(lengths_to_test), max_length, kv_heads, 8),
                )
                lengths = jnp.asarray(lengths_to_test, dtype=jnp.int32)
                positions = jnp.broadcast_to(
                    jnp.arange(max_length, dtype=jnp.int32)[None, :],
                    (len(lengths_to_test), max_length),
                )
                active = positions < lengths[:, None]

                num_blocks = len(lengths_to_test) * 3
                permutation = jax.random.permutation(
                    jax.random.key(999),
                    num_blocks,
                ).reshape(len(lengths_to_test), 3)
                key_pool = jnp.zeros(
                    (num_blocks, block_size, kv_heads, 8),
                )
                value_pool = jnp.zeros_like(key_pool)
                key_pool = write_paged_kv(
                    key_pool,
                    dense_key,
                    permutation,
                    positions,
                    active,
                    block_size,
                )
                value_pool = write_paged_kv(
                    value_pool,
                    dense_value,
                    permutation,
                    positions,
                    active,
                    block_size,
                )
                actual = paged_attention(
                    query,
                    key_pool,
                    value_pool,
                    permutation,
                    lengths,
                    positions,
                    active,
                    block_size=block_size,
                )

                repeated_key = jnp.repeat(
                    dense_key,
                    query_heads // kv_heads,
                    axis=2,
                )
                repeated_value = jnp.repeat(
                    dense_value,
                    query_heads // kv_heads,
                    axis=2,
                )
                causal_mask = (
                    jnp.arange(max_length)[None, None, :]
                    <= positions[:, :, None]
                ) & (
                    jnp.arange(max_length)[None, None, :]
                    < lengths[:, None, None]
                )
                expected = jax.nn.dot_product_attention(
                    query,
                    repeated_key,
                    repeated_value,
                    mask=causal_mask[:, None],
                )
                error = jnp.max(
                    jnp.abs(actual[active] - expected[active])
                )
                self.assertLess(float(error), 2e-5)

    def test_bfloat16_tolerance(self):
        block_size = 4
        lengths = jnp.asarray([1, 5, 10], dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(12)[None, :], (3, 12))
        active = positions < lengths[:, None]
        block_table = jnp.asarray(
            [[4, 1, 8], [2, 7, 0], [6, 3, 5]],
            dtype=jnp.int32,
        )
        q_key, k_key, v_key = jax.random.split(jax.random.key(5), 3)
        query = jax.random.normal(
            q_key, (3, 12, 4, 8), dtype=jnp.bfloat16
        )
        dense_key = jax.random.normal(
            k_key, (3, 12, 2, 8), dtype=jnp.bfloat16
        )
        dense_value = jax.random.normal(
            v_key, (3, 12, 2, 8), dtype=jnp.bfloat16
        )
        key_pool = jnp.zeros((9, 4, 2, 8), dtype=jnp.bfloat16)
        value_pool = jnp.zeros_like(key_pool)
        key_pool = write_paged_kv(
            key_pool, dense_key, block_table, positions, active, block_size
        )
        value_pool = write_paged_kv(
            value_pool, dense_value, block_table, positions, active, block_size
        )
        actual = paged_attention(
            query,
            key_pool,
            value_pool,
            block_table,
            lengths,
            positions,
            active,
            block_size=block_size,
        )
        repeated_key = jnp.repeat(dense_key, 2, axis=2)
        repeated_value = jnp.repeat(dense_value, 2, axis=2)
        mask = (
            jnp.arange(12)[None, None, :] <= positions[:, :, None]
        ) & (
            jnp.arange(12)[None, None, :] < lengths[:, None, None]
        )
        expected = jax.nn.dot_product_attention(
            query,
            repeated_key,
            repeated_value,
            mask=mask[:, None],
        )
        error = jnp.max(
            jnp.abs(
                actual[active].astype(jnp.float32)
                - expected[active].astype(jnp.float32)
            )
        )
        self.assertLessEqual(float(error), 0.0078125)


if __name__ == "__main__":
    unittest.main()
