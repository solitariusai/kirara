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

"""Tests for batched prefill and decode scheduling invariants."""

import unittest

import jax
import jax.numpy as jnp

from kirara import LLM, SamplingParams
from kirara.engine import Request, RequestStatus
from kirara.inputs import NormalizedInput
from kirara.state import write_paged_kv


class _Config:
    num_hidden_layers = 1
    num_attention_heads = 1
    num_key_value_heads = 1
    hidden_size = 1
    head_dim = 1
    dtype = jnp.float32


class _CacheWritingModel:
    """Small JAX model that exposes cache and logit-selection mistakes."""

    config = _Config()
    vocab_size = 64

    class tokenizer:
        eos_token_id = None

        @staticmethod
        def encode(text):
            return [ord(char) % 64 for char in text]

        @staticmethod
        def decode(token_ids):
            return " ".join(map(str, token_ids))

    def paged_forward(
        self,
        *,
        input_ids,
        position_ids,
        token_active,
        block_table,
        sequence_lengths,
        key_pool,
        value_pool,
        logit_indices,
        block_size,
    ):
        del sequence_lengths
        updates = input_ids.astype(jnp.float32)[:, :, None, None]

        def update_layer(layer_pool):
            return write_paged_kv(
                layer_pool,
                updates,
                block_table,
                position_ids,
                token_active,
                block_size,
            )

        key_pool = jax.vmap(update_layer)(key_pool)
        value_pool = jax.vmap(update_layer)(value_pool)
        all_logits = jax.nn.one_hot(
            (input_ids + 1) % self.vocab_size,
            self.vocab_size,
            dtype=jnp.float32,
        )
        logits = jnp.take_along_axis(
            all_logits,
            logit_indices[:, None, None],
            axis=1,
        )
        return logits, (key_pool, value_pool)

    def __call__(
        self,
        input_ids,
        *,
        kv_cache,
        cache_position,
        logits_to_keep,
        **_,
    ):
        updates = input_ids.astype(jnp.float32)[:, :, None, None]

        def update_layer(cache_layer):
            return jax.vmap(
                lambda row, values, positions: row.at[positions].set(values)
            )(cache_layer, updates, cache_position)

        key = jax.vmap(update_layer)(kv_cache[0])
        value = jax.vmap(update_layer)(kv_cache[1])
        all_logits = jax.nn.one_hot(
            (input_ids + 1) % self.vocab_size,
            self.vocab_size,
            dtype=jnp.float32,
        )
        if isinstance(logits_to_keep, int):
            logits = all_logits[:, -logits_to_keep:]
        else:
            indices = jnp.asarray(logits_to_keep, dtype=jnp.int32)
            logits = jnp.take_along_axis(
                all_logits,
                indices[:, None, None],
                axis=1,
            )
        return SimpleNamespace(logits=logits, kv_cache=(key, value))


def _request(request_id, tokens, slot_id):
    return Request(
        request_id=str(request_id),
        inputs=NormalizedInput(input_ids=tokens),
        sampling_params=SamplingParams(
            temperature=0,
            max_tokens=1,
        ),
        slot_id=slot_id,
    )


class BatchedPrefillTest(unittest.TestCase):
    def make_scheduler(self, **kwargs):
        block_size = kwargs.pop("block_size", 16)
        max_batched_tokens = kwargs.pop("max_num_batched_tokens", 128)
        llm = LLM(
            _CacheWritingModel(),
            model_impl="taktiny",
            max_num_seqs=4,
            max_model_len=64,
            max_num_batched_tokens=max_batched_tokens,
            max_num_prefill_tokens=128,
            block_size=block_size,
            **kwargs,
        )
        return llm.scheduler

    def test_lengths_reuse_three_static_buckets(self):
        scheduler = self.make_scheduler()
        lengths = [5, 6, 7, 8, 16, 31, 32, 33]
        expected = [16, 16, 16, 16, 16, 32, 32, 64]

        self.assertEqual(
            [scheduler._get_bucket_for_len(length) for length in lengths],
            expected,
        )
        for length in lengths:
            scheduler.generate(
                [[2] * length],
                SamplingParams(temperature=0, max_tokens=1),
            )

        self.assertEqual(scheduler.compiled_prefill_buckets, {16, 32, 64})
        self.assertEqual(len(scheduler._compiled_prefill_steps), 3)

    def test_four_requests_share_one_prefill_invocation(self):
        scheduler = self.make_scheduler()
        prompts = [[2] * length for length in (10, 15, 20, 25)]
        sampling = SamplingParams(temperature=0, max_tokens=1)
        requests = [
            scheduler.add_request(prompt, sampling) for prompt in prompts
        ]

        scheduler.step()

        self.assertEqual(scheduler.prefill_model_invocations, 1)
        self.assertEqual(scheduler.prefill_bucket_history, [32])
        self.assertEqual([request.slot_id for request in requests], [0, 1, 2, 3])

    def test_total_batch_token_budget_limits_prefill_admission(self):
        scheduler = self.make_scheduler(max_num_batched_tokens=64)
        sampling = SamplingParams(temperature=0, max_tokens=1)
        requests = [
            scheduler.add_request([2] * length, sampling)
            for length in (10, 15, 20, 25)
        ]

        scheduler.step()

        self.assertEqual(scheduler.prefill_model_invocations, 1)
        self.assertEqual(
            [request.status for request in requests[:3]],
            [RequestStatus.FINISHED] * 3,
        )
        self.assertIs(requests[3].status, RequestStatus.WAITING)

    def test_padding_cannot_mutate_cache_or_select_padding_logits(self):
        scheduler = self.make_scheduler()
        inactive_block = scheduler.cache_manager.ensure_position(1, 0)
        key, value = scheduler.cache_manager.cache
        key = key.at[:, inactive_block].set(7)
        value = value.at[:, inactive_block].set(9)
        scheduler.cache_manager.cache = (key, value)

        requests = [
            _request(0, [3, 4, 5, 6, 7], 2),
            _request(1, [10, 11, 12, 13, 14, 15, 16, 17], 3),
        ]
        scheduler._run_batched_prefill(requests)
        key, value = map(jax.device_get, scheduler.cache_manager.cache)
        first_block = scheduler.cache_manager.slot_blocks(2)[0]
        second_block = scheduler.cache_manager.slot_blocks(3)[0]

        self.assertTrue(jnp.all(key[:, inactive_block] == 7))
        self.assertTrue(jnp.all(value[:, inactive_block] == 9))
        self.assertTrue(jnp.all(key[:, first_block, 5:] == 0))
        self.assertTrue(jnp.all(value[:, first_block, 5:] == 0))
        self.assertTrue(jnp.all(key[:, second_block, 8:] == 0))
        self.assertTrue(jnp.all(value[:, second_block, 8:] == 0))
        self.assertEqual(
            [request.generated_tokens for request in requests],
            [[8], [18]],
        )
        self.assertEqual([request.position for request in requests], [5, 8])
        self.assertEqual([request.slot_id for request in requests], [2, 3])

    def test_decode_batches_active_slots_and_preserves_inactive_cache(self):
        scheduler = self.make_scheduler()
        inactive_zero = scheduler.cache_manager.ensure_position(0, 0)
        active_one = scheduler.cache_manager.ensure_position(1, 5)
        inactive_two = scheduler.cache_manager.ensure_position(2, 0)
        active_three = scheduler.cache_manager.ensure_position(3, 8)
        key, value = scheduler.cache_manager.cache
        key = key.at[:, inactive_zero].set(7)
        value = value.at[:, inactive_zero].set(8)
        key = key.at[:, inactive_two].set(9)
        value = value.at[:, inactive_two].set(10)
        scheduler.cache_manager.cache = (key, value)

        first = _request(0, [1, 2], 1)
        first.generated_tokens = [20]
        first.position = 5
        first.status = RequestStatus.DECODING
        second = _request(1, [3, 4], 3)
        second.generated_tokens = [30]
        second.position = 8
        second.status = RequestStatus.DECODING
        scheduler.slots[1] = first
        scheduler.slots[3] = second

        scheduler._run_decode([1, 3])
        key, value = map(jax.device_get, scheduler.cache_manager.cache)

        self.assertEqual(scheduler.decode_model_invocations, 1)
        self.assertTrue(scheduler.decode_compiled)
        self.assertTrue(jnp.all(key[:, inactive_zero] == 7))
        self.assertTrue(jnp.all(value[:, inactive_zero] == 8))
        self.assertTrue(jnp.all(key[:, inactive_two] == 9))
        self.assertTrue(jnp.all(value[:, inactive_two] == 10))
        self.assertTrue(jnp.all(key[:, active_one, 5] == 20))
        self.assertTrue(jnp.all(value[:, active_one, 5] == 20))
        self.assertTrue(jnp.all(key[:, active_three, 8] == 30))
        self.assertTrue(jnp.all(value[:, active_three, 8] == 30))
        self.assertEqual(first.generated_tokens, [20, 21])
        self.assertEqual(second.generated_tokens, [30, 31])
        self.assertEqual([first.position, second.position], [6, 9])
        self.assertEqual([first.slot_id, second.slot_id], [1, 3])

    def test_decode_uses_one_invocation_per_step_not_per_request(self):
        scheduler = self.make_scheduler()
        results = scheduler.generate(
            [[2] * length for length in (5, 6, 7, 8)],
            SamplingParams(temperature=0, max_tokens=3),
        )

        self.assertEqual(scheduler.prefill_model_invocations, 1)
        self.assertEqual(scheduler.decode_model_invocations, 2)
        self.assertEqual(scheduler.compiled_executable_count, 2)
        self.assertEqual(results, [[3, 4, 5]] * 4)

    def test_dynamic_lifecycle_reuses_slot_and_block_without_recompile(self):
        scheduler = self.make_scheduler(block_size=4)
        short = scheduler.add_request(
            [2] * 4,
            SamplingParams(
                temperature=0,
                max_tokens=2,
                stop_token_ids=[],
            ),
        )
        long = scheduler.add_request(
            [3] * 4,
            SamplingParams(
                temperature=0,
                max_tokens=6,
                stop_token_ids=[],
            ),
        )

        scheduler.step()
        self.assertIs(short.status, RequestStatus.FINISHED)
        self.assertEqual(long.slot_id, 1)
        self.assertEqual(scheduler.cache_manager.slot_blocks(0), ())

        replacement = scheduler.add_request(
            [4] * 4,
            SamplingParams(
                temperature=0,
                max_tokens=4,
                stop_token_ids=[],
            ),
        )
        scheduler.step()
        self.assertEqual(replacement.slot_id, 0)
        self.assertEqual(scheduler.cache_manager.slot_blocks(0)[0], 0)
        self.assertEqual(long.slot_id, 1)

        while scheduler.step():
            pass

        self.assertIs(long.status, RequestStatus.FINISHED)
        self.assertIs(replacement.status, RequestStatus.FINISHED)
        self.assertEqual(scheduler.compiled_prefill_buckets, {16})
        self.assertTrue(scheduler.decode_compiled)
        self.assertEqual(scheduler.compiled_executable_count, 2)
        self.assertEqual(scheduler.cache_manager.free_block_count, 64)


if __name__ == "__main__":
    unittest.main()
