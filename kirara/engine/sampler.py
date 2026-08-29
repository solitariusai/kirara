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

"""Request-specific token sampling from model logits."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kirara.sampling import SamplingParams


class Sampler:
    """Convert model logits to request-specific token IDs."""

    def __init__(self, seed: int = 0) -> None:
        """Initialize the sampler with a random seed.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
        """
        self.seed = seed

    def sample(
        self,
        logits: jax.Array,
        rows: list[int],
        params: list[SamplingParams],
        steps: list[int],
    ) -> list[int]:
        """Sample token IDs from the given logits for a batch of requests.

        Args:
            logits (jax.Array): The model output logits to sample from.
            rows (list[int]): The row indices corresponding to each request in the logits array.
            params (list[SamplingParams]): Sampling configuration for each request.
            steps (list[int]): The current generation step for each request, used for PRNG folding.

        Returns:
            list[int]: The sampled token IDs, one for each request.
        """
        tokens: list[int] = []
        for row, sampling, step in zip(rows, params, steps, strict=True):
            row_logits = logits[row]
            if sampling.temperature == 0:
                tokens.append(int(jnp.argmax(row_logits)))
                continue
            scores = row_logits / sampling.temperature
            if sampling.top_k:
                top_k = min(sampling.top_k, scores.shape[-1])
                cutoff = jnp.sort(scores)[-top_k]
                scores = jnp.where(scores < cutoff, -jnp.inf, scores)
            if sampling.top_p < 1:
                order = jnp.argsort(scores)[::-1]
                ordered = scores[order]
                cumulative = jnp.cumsum(jax.nn.softmax(ordered))
                remove = cumulative - jax.nn.softmax(ordered) >= sampling.top_p
                ordered = jnp.where(remove, -jnp.inf, ordered)
                scores = jnp.full_like(scores, -jnp.inf).at[order].set(ordered)
            seed = sampling.seed if sampling.seed is not None else self.seed
            key = jax.random.fold_in(jax.random.key(seed), step)
            key = jax.random.fold_in(key, row)
            tokens.append(int(jax.random.categorical(key, scores)))
        return tokens


__all__ = ["Sampler"]
