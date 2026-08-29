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

"""Tests for Kirara's runtime ownership boundaries."""

import inspect
import unittest

import jax
import jax.numpy as jnp

from kirara import LLM
from kirara.engine import Runner, Scheduler
from kirara.inputs import InputPart, TokenizerInputProcessor
from kirara.models import (
    Capabilities,
    LoadedModel,
    Config,
    Output,
    model_registry,
)
from kirara.sharding import build_mesh
from kirara.state import State
from kirara.types import ExecutionPhase, MeshSpec, ModelSource


class _Tokenizer:
    def encode(self, text):
        return [len(text), 1]

    def decode(self, token_ids):
        return str(token_ids)


class _Loader:
    def load(self, source, **kwargs):
        del kwargs
        return LoadedModel(model=source, tokenizer=_Tokenizer())


class _EncoderAdapter:
    config = Config(max_model_len=16, dtype=jnp.float32)
    capabilities = Capabilities(encode=True)
    state_spec = None
    tokenizer = _Tokenizer()

    def __call__(self, batch, state=None):
        del state
        embeddings = jnp.sum(batch.input_ids, axis=-1, keepdims=True)
        return Output(embeddings=embeddings)


def _adapter(loaded, attention_backend, runtime):
    del loaded, attention_backend, runtime
    return _EncoderAdapter()


def _processor(loaded):
    return TokenizerInputProcessor(loaded.tokenizer)


class ArchitectureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_registry.register(
            "test-encoder",
            loader=_Loader(),
            adapter=_adapter,
            processor=_processor,
            detect=lambda source: False,
        )

    def test_state_supports_mapping_attributes_and_pytree(self):
        state = State(kv=jnp.asarray([1, 2]), recurrent=jnp.asarray([3]))

        self.assertTrue("kv" in state)
        self.assertEqual(state.kv.tolist(), [1, 2])
        self.assertEqual(state["recurrent"].tolist(), [3])
        self.assertEqual(list(state.keys()), ["kv", "recurrent"])
        leaves, tree = jax.tree_util.tree_flatten(state)
        restored = jax.tree_util.tree_unflatten(tree, leaves)
        self.assertEqual(restored.kv.tolist(), [1, 2])

    def test_runner_not_scheduler_owns_jit(self):
        self.assertNotIn("jax.jit", inspect.getsource(Scheduler))
        self.assertIn("jax.jit", inspect.getsource(Runner))

    def test_registered_encoder_uses_same_llm_facade(self):
        llm = LLM(
            object(),
            model_impl="test-encoder",
            max_num_seqs=2,
        )

        embeddings = llm.encode(["hi", "kirara"])

        self.assertEqual(embeddings.tolist(), [[3], [7]])
        self.assertEqual(llm.runner.encode_model_invocations, 1)

    def test_existing_adapter_instance_is_auto_detected(self):
        llm = LLM(_EncoderAdapter(), max_num_seqs=2)

        embeddings = llm.encode(["a", "abcd"])

        self.assertEqual(embeddings.tolist(), [[2], [5]])
        self.assertEqual(llm.adapter.capabilities, Capabilities(encode=True))

    def test_single_device_mesh_is_jax_mesh(self):
        mesh = build_mesh({"data": 1})

        self.assertEqual(mesh.axis_names, ("data",))

    def test_shared_and_domain_types_are_importable(self):
        self.assertIs(ModelSource.__value__, object)
        self.assertIsNotNone(MeshSpec)
        self.assertIsNotNone(ExecutionPhase)
        self.assertEqual(InputPart.__required_keys__, frozenset({"type"}))


if __name__ == "__main__":
    unittest.main()
