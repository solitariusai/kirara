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

"""Tests for Kirara's public API."""

import unittest

import jax
import jax.numpy as jnp

import kirara
from kirara import LLM, SamplingParams


class _Config:
    num_hidden_layers = 1
    num_attention_heads = 1
    num_key_value_heads = 1
    hidden_size = 1
    head_dim = 1
    max_position_embeddings = 64
    dtype = jnp.float32


class _Tokenizer:
    eos_token_id = None

    def encode(self, text):
        return [len(text), 2]

    def decode(self, token_ids):
        return " ".join(str(token) for token in token_ids)


class _Model:
    config = _Config()
    tokenizer = _Tokenizer()
    vocab_size = 64

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
        del position_ids
        del token_active
        del block_table
        del sequence_lengths
        del block_size
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


class PublicApiTest(unittest.TestCase):
    def test_public_api_is_small(self):
        self.assertEqual(kirara.__all__, ["LLM", "SamplingParams"])

    def test_model_instance_generates_structured_outputs(self):
        llm = LLM(
            _Model(),
            model_impl="taktiny",
            max_num_seqs=4,
            max_model_len=64,
            max_num_prefill_tokens=128,
        )

        outputs = llm.generate(
            ["hi", "kirara"],
            SamplingParams(temperature=0, max_tokens=2),
        )

        self.assertEqual([output.prompt for output in outputs], ["hi", "kirara"])
        self.assertEqual(
            [output.prompt_ids for output in outputs],
            [[2, 2], [6, 2]],
        )
        self.assertEqual(
            [output.output[0].token_ids for output in outputs],
            [[3, 4], [3, 4]],
        )
        self.assertEqual(
            [output.output[0].text for output in outputs],
            ["3 4", "3 4"],
        )
        self.assertEqual(llm.runner.prefill_model_invocations, 1)
        self.assertEqual(llm.runner.decode_model_invocations, 1)

    def test_non_multimodal_model_rejects_structured_image(self):
        llm = LLM(
            _Model(),
            max_num_seqs=1,
            max_model_len=64,
        )

        with self.assertRaisesRegex(ValueError, "multimodal"):
            llm.generate(
                [[
                    {"type": "text", "text": "describe"},
                    {"type": "image", "image": object()},
                ]],
                SamplingParams(temperature=0, max_tokens=1),
            )

    def test_unknown_explicit_implementation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown model implementation"):
            LLM(_Model(), model_impl="missing")

    def test_sampling_params_validate_runtime_values(self):
        with self.assertRaises(ValueError):
            SamplingParams(max_tokens=0)
        with self.assertRaises(ValueError):
            SamplingParams(top_p=0)


if __name__ == "__main__":
    unittest.main()
