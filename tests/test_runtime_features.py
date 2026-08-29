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

"""Correctness gates for completed optional runtime paths."""

from unittest import TestCase, mock

import jax
import jax.numpy as jnp
import qwix

from kirara import LLM, SamplingParams
from kirara.engine import RequestStatus
from kirara.inputs import NormalizedInput, TokenizerInputProcessor
from kirara.integrations.qwix import validate_qwix_config
from kirara.integrations.taktiny.loader import TakTinyLoader
from kirara.models import Capabilities, Config, Output, StateSpec
from kirara.state import State, write_paged_kv


class _Tokenizer:
    eos_token_id = None

    @staticmethod
    def encode(text):
        return [int(text) if text.isdigit() else len(text)]

    @staticmethod
    def decode(token_ids):
        return " ".join(map(str, token_ids))


class _PagedConfig:
    num_hidden_layers = 1
    num_attention_heads = 1
    num_key_value_heads = 1
    hidden_size = 1
    head_dim = 1
    dtype = jnp.float32


class _PagedModel:
    config = _PagedConfig()
    tokenizer = _Tokenizer()
    vocab_size = 128

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
        key_pool = jax.vmap(
            lambda pool: write_paged_kv(
                pool,
                updates,
                block_table,
                position_ids,
                token_active,
                block_size,
            )
        )(key_pool)
        value_pool = jax.vmap(
            lambda pool: write_paged_kv(
                pool,
                updates,
                block_table,
                position_ids,
                token_active,
                block_size,
            )
        )(value_pool)
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


class _RecurrentAdapter:
    config = Config(max_model_len=32, dtype=jnp.float32)
    capabilities = Capabilities(generate=True, stateful=True)
    state_spec = StateSpec(
        kinds=("recurrent", "convolution"),
        shapes={"recurrent": (2,), "convolution": (3,)},
    )
    tokenizer = _Tokenizer()
    processor = TokenizerInputProcessor(tokenizer)
    vocab_size = 128

    def __call__(self, batch, state=None):
        assert state is not None
        active_values = jnp.where(
            batch.active_mask,
            batch.input_ids,
            0,
        ).sum(axis=1, keepdims=True)
        recurrent = state.recurrent + active_values
        convolution = state.convolution + active_values
        all_logits = jax.nn.one_hot(
            (batch.input_ids + 1) % self.vocab_size,
            self.vocab_size,
            dtype=jnp.float32,
        )
        indices = batch.metadata["logit_indices"]
        logits = jnp.take_along_axis(
            all_logits,
            indices[:, None, None],
            axis=1,
        )
        return Output(
            logits=logits,
            state=State(
                recurrent=recurrent,
                convolution=convolution,
            ),
        )


class _MultimodalProcessor:
    def __call__(self, inputs):
        rows = inputs if isinstance(inputs, list) else [inputs]
        normalized = []
        for row in rows:
            modalities = {}
            token = 2
            for part in row:
                if part["type"] == "text":
                    token = int(part["text"])
                elif part["type"] == "image":
                    modalities["image"] = jnp.asarray(part["image"])
            normalized.append(
                NormalizedInput(
                    input_ids=[token],
                    prompt=str(token),
                    modalities=modalities,
                )
            )
        return normalized

    def decode(self, token_ids):
        return " ".join(map(str, token_ids))


class _MultimodalAdapter:
    config = Config(max_model_len=16, dtype=jnp.float32)
    capabilities = Capabilities(generate=True, multimodal=True)
    state_spec = None
    tokenizer = _Tokenizer()
    processor = _MultimodalProcessor()
    vocab_size = 64

    def __call__(self, batch, state=None):
        del state
        image = batch.modalities["image"]
        image_active = batch.modality_mask["image"]
        image_bias = image.reshape(image.shape[0], -1).sum(axis=1)
        image_bias = jnp.where(image_active, image_bias, 0).astype(jnp.int32)
        next_ids = (
            batch.input_ids + image_bias[:, None] + 1
        ) % self.vocab_size
        all_logits = jax.nn.one_hot(
            next_ids,
            self.vocab_size,
            dtype=jnp.float32,
        )
        indices = batch.metadata["logit_indices"]
        logits = jnp.take_along_axis(
            all_logits,
            indices[:, None, None],
            axis=1,
        )
        return Output(logits=logits)


class RuntimeFeatureTest(TestCase):
    def test_chunked_prefill_advances_real_ranges_and_reuses_one_bucket(self):
        llm = LLM(
            _PagedModel(),
            model_impl="taktiny",
            max_num_seqs=2,
            max_model_len=32,
            max_num_batched_tokens=4,
            max_num_prefill_tokens=4,
            block_size=4,
        )
        request = llm.scheduler.add_request(
            [2] * 10,
            SamplingParams(temperature=0, max_tokens=1),
        )

        llm.scheduler.step()
        self.assertIs(request.status, RequestStatus.PREFILL)
        self.assertEqual(request.prefill_position, 4)
        llm.scheduler.step()
        self.assertEqual(request.prefill_position, 8)
        key_pool = llm.state_manager.paged_cache.cache[0]
        blocks = llm.state_manager.paged_cache.slot_blocks(0)
        self.assertTrue(jnp.all(key_pool[:, blocks[0]] == 2))
        self.assertTrue(jnp.all(key_pool[:, blocks[1]] == 2))
        llm.scheduler.step()

        self.assertIs(request.status, RequestStatus.FINISHED)
        self.assertEqual(request.generated_tokens, [3])
        self.assertEqual(llm.scheduler.prefill_token_history, [4, 4, 2])
        self.assertEqual(llm.runner.prefill_model_invocations, 3)
        self.assertEqual(llm.runner.compiled_prefill_buckets, {16})

    def test_chunked_prefill_can_be_disabled(self):
        llm = LLM(
            _PagedModel(),
            model_impl="taktiny",
            max_num_seqs=1,
            max_model_len=32,
            max_num_prefill_tokens=4,
            enable_chunked_prefill=False,
        )
        llm.scheduler.add_request([2] * 5)
        with self.assertRaisesRegex(ValueError, "prefill token budget"):
            llm.scheduler.step()

    def test_prefix_cache_reuses_authoritative_pages(self):
        llm = LLM(
            _PagedModel(),
            model_impl="taktiny",
            max_num_seqs=1,
            max_model_len=32,
            block_size=4,
            enable_prefix_caching=True,
        )
        prompt = list(range(1, 10))
        params = SamplingParams(temperature=0, max_tokens=1)

        first = llm.scheduler.generate([prompt], params)
        cache = llm.state_manager.paged_cache
        self.assertEqual(first, [[10]])
        self.assertEqual(len(llm.state_manager.prefix_cache), 1)
        self.assertEqual(cache.free_block_count, cache.num_blocks - 2)

        second = llm.scheduler.generate([prompt], params)
        self.assertEqual(second, [[10]])
        self.assertEqual(llm.state_manager.prefix_cache.hits, 1)
        self.assertEqual(llm.scheduler.prefill_token_history, [9, 1])
        self.assertEqual(llm.runner.compiled_prefill_buckets, {16})
        self.assertEqual(cache.free_block_count, cache.num_blocks - 2)
        self.assertEqual(
            sum(cache.block_refcount(i) for i in range(cache.num_blocks)),
            2,
        )
        llm.state_manager.prefix_cache.clear()
        self.assertEqual(cache.free_block_count, cache.num_blocks)
        self.assertTrue(jnp.all(cache.cache[0] == 0))
        self.assertTrue(jnp.all(cache.cache[1] == 0))

    def test_recurrent_state_is_slot_managed_and_recycled(self):
        llm = LLM(
            _RecurrentAdapter(),
            max_num_seqs=2,
            max_model_len=32,
        )
        request = llm.scheduler.add_request(
            [2, 3],
            SamplingParams(
                temperature=0,
                max_tokens=3,
                stop_token_ids=[],
            ),
        )

        llm.scheduler.step()
        self.assertIs(request.status, RequestStatus.DECODING)
        state = llm.state_manager.state
        self.assertTrue(jnp.all(state.recurrent[0] == 9))
        self.assertTrue(jnp.all(state.convolution[0] == 9))
        self.assertTrue(jnp.all(state.recurrent[1] == 0))
        while llm.scheduler.step():
            pass
        self.assertTrue(jnp.all(llm.state_manager.state.recurrent == 0))
        self.assertTrue(jnp.all(llm.state_manager.state.convolution == 0))

        with self.assertRaisesRegex(ValueError, "pure paged KV"):
            LLM(
                _RecurrentAdapter(),
                max_num_seqs=1,
                enable_prefix_caching=True,
            )

    def test_multimodal_tensors_and_masks_reach_one_batched_call(self):
        llm = LLM(
            _MultimodalAdapter(),
            max_num_seqs=2,
            max_model_len=16,
        )
        outputs = llm.generate(
            [
                [
                    {"type": "text", "text": "2"},
                    {"type": "image", "image": [1.0, 2.0]},
                ],
                [{"type": "text", "text": "2"}],
            ],
            SamplingParams(temperature=0, max_tokens=2),
        )

        self.assertEqual(
            [output.output[0].token_ids for output in outputs],
            [[6, 10], [3, 4]],
        )
        self.assertEqual(llm.runner.prefill_model_invocations, 1)
        self.assertEqual(llm.runner.decode_model_invocations, 1)

    def test_qwix_rules_are_validated_and_forwarded_unchanged(self):
        rule = qwix.QuantizationRule(weight_qtype=jnp.int8)
        self.assertIs(validate_qwix_config(rule), rule)
        with self.assertRaises(TypeError):
            validate_qwix_config(object())

        loader = TakTinyLoader()
        tokenizer = object()
        model = object()
        with (
            mock.patch(
                "kirara.integrations.taktiny.loader.Maestro.from_pretrained",
                return_value=model,
            ) as load,
            mock.patch.object(
                loader,
                "_load_tokenizer",
                return_value=tokenizer,
            ),
        ):
            loaded = loader.load(
                "repo/model",
                dtype=jnp.bfloat16,
                revision=None,
                tokenizer=None,
                quantization=rule,
                mesh=None,
                trust_remote_code=False,
                model_kwargs={},
            )
        self.assertIs(load.call_args.kwargs["quant"], rule)
        self.assertEqual(loaded.metadata["quantization"], "qwix")
        with self.assertRaisesRegex(ValueError, "load-time"):
            loader.load(
                _PagedModel(),
                dtype=jnp.float32,
                revision=None,
                tokenizer=None,
                quantization=rule,
                mesh=None,
                trust_remote_code=False,
                model_kwargs={},
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
