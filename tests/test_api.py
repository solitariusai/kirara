import unittest

import jax
import jax.numpy as jnp

from kirara import LLM, Output, OutputSampling, SamplingParams, XLLM


class _Config:
    num_hidden_layers = 1
    num_attention_heads = 1
    num_key_value_heads = 1
    hidden_size = 1
    head_dim = 1
    dtype = jnp.float32


class _NativeModel:
    config = _Config()
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


class _ExternalModel:
    def __init__(self):
        self.calls = []

    def generate(self, input_ids, sampling_params):
        self.calls.append((input_ids, sampling_params))
        return [
            [row[-1] + offset for offset in range(1, 3)]
            for row in input_ids
        ]


class _Tokenizer:
    eos_token_id = 99

    def decode(self, token_ids):
        return " ".join(str(token) for token in token_ids)


class PublicApiTest(unittest.TestCase):
    def test_public_exports_are_distinct_model_paths(self):
        self.assertIsNot(XLLM, LLM)
        self.assertTrue(Output)
        self.assertTrue(OutputSampling)

    def test_native_model_uses_kirara_scheduler(self):
        llm = LLM(_NativeModel(), max_seq_len=16)
        result = llm([3, 4, 5], SamplingParams(max_new_tokens=2))

        self.assertFalse(llm.uses_external_model)
        self.assertIsNotNone(llm.scheduler)
        self.assertEqual(result[0].output[0].token_ids, [6, 7])
        self.assertEqual(llm.scheduler.prefill_model_invocations, 1)
        self.assertEqual(llm.scheduler.decode_model_invocations, 1)

    def test_external_model_uses_default_generator(self):
        external = _ExternalModel()
        llm = LLM(
            XLLM(external),
            tokenizer=_Tokenizer(),
            tokenize_fn=lambda text: [len(text), 2],
        )
        sampling = SamplingParams(max_new_tokens=2)

        results = llm(["hi", "kirara"], sampling)

        self.assertTrue(llm.uses_external_model)
        self.assertIsNone(llm.scheduler)
        self.assertEqual(len(external.calls), 1)
        self.assertEqual(external.calls[0], ([[2, 2], [6, 2]], sampling))
        self.assertEqual(
            [result.output[0].token_ids for result in results],
            [[3, 4], [3, 4]],
        )
        self.assertEqual(
            [result.output[0].text for result in results],
            ["3 4", "3 4"],
        )

    def test_external_model_can_use_explicit_adapter(self):
        adapter = XLLM(
            object(),
            generate_fn=lambda rows, params: [
                [params.max_new_tokens] for _ in rows
            ],
        )
        result = LLM(adapter)(
            [[3, 4], [5, 6]],
            SamplingParams(max_new_tokens=7),
        )

        self.assertEqual(
            [item.output[0].token_ids for item in result],
            [[7], [7]],
        )

    def test_invalid_native_model_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "must expose config"):
            LLM(object())

    def test_text_without_tokenizer_is_rejected(self):
        llm = LLM(_NativeModel(), max_seq_len=16)

        with self.assertRaisesRegex(ValueError, "tokenizer or tokenize_fn"):
            llm("hello", SamplingParams(max_new_tokens=1))


if __name__ == "__main__":
    unittest.main()
