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

"""Input-to-output orchestration for generation and encoding."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kirara.engine.output import Output, OutputSampling
from kirara.engine.runner import Runner
from kirara.engine.scheduler import Scheduler
from kirara.inputs import GenerationInput, InputProcessor, NormalizedInput
from kirara.models import Adapter, Batch
from kirara.sampling import SamplingParams


class Engine:
    """Coordinate input processing with generation or encoding execution."""

    def __init__(
        self,
        model: Adapter,
        processor: InputProcessor,
        runner: Runner,
        scheduler: Scheduler,
    ) -> None:
        """Initialize the Engine.

        Args:
            model (Adapter): The model adapter.
            processor (InputProcessor): The input processor.
            runner (Runner): The execution runner.
            scheduler (Scheduler): The scheduler.
        """
        self.model = model
        self.processor = processor
        self.runner = runner
        self.scheduler = scheduler

    def generate(
        self,
        inputs: GenerationInput,
        sampling_params: SamplingParams | None = None,
    ) -> list[Output]:
        """Generate text from the given inputs.

        Args:
            inputs (GenerationInput): The input prompts to generate from.
            sampling_params (SamplingParams | None, optional): The sampling parameters. Defaults to None.

        Raises:
            ValueError: If the model does not support generation.
            NotImplementedError: If text stop sequences are provided.

        Returns:
            list[Output]: A list of generation outputs.
        """
        if not self.model.capabilities.generate:
            raise ValueError("model does not support generation")
        normalized = self.processor(inputs)
        self._validate_modalities(normalized)
        params = sampling_params or SamplingParams()
        if params.stop is not None:
            raise NotImplementedError(
                "text stop sequences are not implemented; use stop_token_ids"
            )
        generated = self.scheduler.generate(normalized, params)
        return [
            Output(
                prompt=item.prompt,
                prompt_ids=list(item.input_ids),
                output=[
                    OutputSampling(
                        text=self.processor.decode(token_ids),
                        token_ids=token_ids,
                        num_tokens=len(token_ids),
                        stop_reason=self._stop_reason(token_ids, params),
                    )
                ],
            )
            for item, token_ids in zip(normalized, generated, strict=True)
        ]

    def encode(self, inputs: GenerationInput) -> jax.Array:
        """Encode the given inputs into embeddings.

        Args:
            inputs (GenerationInput): The inputs to encode.

        Raises:
            ValueError: If the model does not support encoding.
            RuntimeError: If the encode adapter did not return embeddings.

        Returns:
            jax.Array: The computed embeddings.
        """
        if not self.model.capabilities.encode:
            raise ValueError("model does not support encoding")
        normalized = self.processor(inputs)
        self._validate_modalities(normalized)
        if not normalized:
            return jnp.empty((0, 0), dtype=jnp.float32)
        max_length = max(len(item.input_ids) for item in normalized)
        bucket = self.runner.bucket_for_length(max_length)
        input_ids = jnp.zeros((len(normalized), bucket), dtype=jnp.int32)
        positions = jnp.broadcast_to(
            jnp.arange(bucket, dtype=jnp.int32)[None, :],
            (len(normalized), bucket),
        )
        active = jnp.zeros((len(normalized), bucket), dtype=jnp.bool_)
        for row, item in enumerate(normalized):
            length = len(item.input_ids)
            input_ids = input_ids.at[row, :length].set(
                jnp.asarray(item.input_ids, dtype=jnp.int32)
            )
            active = active.at[row, :length].set(True)
        output = self.runner.execute(
            Batch(
                input_ids=input_ids,
                positions=positions,
                active_mask=active,
            ),
            phase="encode",
        )
        if output.embeddings is None:
            raise RuntimeError("encode adapter did not return embeddings")
        return output.embeddings

    def _validate_modalities(self, inputs: list[NormalizedInput]) -> None:
        """Validate that the model supports the input modalities.

        Args:
            inputs (list[NormalizedInput]): The normalized inputs to check.

        Raises:
            ValueError: If the inputs contain modalities not supported by the model.
        """
        has_modalities = any(item.modalities for item in inputs)
        if has_modalities and not self.model.capabilities.multimodal:
            raise ValueError("model does not support multimodal inputs")

    def _stop_reason(
        self,
        tokens: list[int],
        params: SamplingParams,
    ) -> str:
        """Determine the reason for stopping generation.

        Args:
            tokens (list[int]): The generated tokens.
            params (SamplingParams): The sampling parameters used.

        Returns:
            str: The stop reason ("stop" or "length").
        """
        if (
            tokens
            and params.stop_token_ids is not None
            and tokens[-1] in params.stop_token_ids
        ):
            return "stop"
        return "length"


__all__ = ["Engine"]
