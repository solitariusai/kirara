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

"""TakTiny repository and model-instance loading."""

from __future__ import annotations

from typing import Any

from jax.sharding import Mesh
from taktiny import Maestro

from kirara.models.loader import LoadedModel
from kirara.types import ModelDType, ModelKwargs, ModelSource


class TakTinyLoader:
    """Load repository sources or accept existing TakTiny model instances."""

    def load(
        self,
        source: ModelSource,
        *,
        dtype: ModelDType,
        revision: str | None,
        tokenizer: str | None,
        quantization: Any,
        mesh: Mesh | None,
        trust_remote_code: bool,
        model_kwargs: ModelKwargs,
    ) -> LoadedModel:
        """Loads a TakTiny model and its tokenizer from a source string or instance.

        Args:
            source (ModelSource): The model source string or instance.
            dtype (ModelDType): The target data type.
            revision (str | None): The model revision.
            tokenizer (str | None): The tokenizer source string.
            quantization (Any): The quantization configuration.
            mesh (Mesh | None): The device mesh for parallelization.
            trust_remote_code (bool): Whether to trust remote code.
            model_kwargs (ModelKwargs): Additional model keyword arguments.

        Raises:
            ValueError: If model_kwargs are provided for an existing model instance.

        Returns:
            LoadedModel: The loaded model wrapper.
        """
        if isinstance(source, str):
            kwargs = dict(model_kwargs)
            if revision is not None:
                kwargs["revision"] = revision
            if trust_remote_code:
                kwargs["trust_remote_code"] = True
            model = Maestro.from_pretrained(
                source,
                dtype=dtype,
                quant=quantization,
                mesh=mesh,
                **kwargs,
            )
            tokenizer_source = tokenizer or source
            tokenizer_instance = self._load_tokenizer(
                tokenizer_source,
                revision,
                trust_remote_code,
            )
            return LoadedModel(
                model=model,
                tokenizer=tokenizer_instance,
                metadata={"source": source},
            )

        if model_kwargs:
            raise ValueError(
                "model_kwargs only apply when loading a model source"
            )
        tokenizer_instance = getattr(source, "tokenizer", None)
        if tokenizer is not None:
            tokenizer_instance = self._load_tokenizer(
                tokenizer,
                revision,
                trust_remote_code,
            )
        return LoadedModel(model=source, tokenizer=tokenizer_instance)

    @staticmethod
    def _load_tokenizer(
        source: str,
        revision: str | None,
        trust_remote_code: bool,
    ) -> Any:
        """Loads a Hugging Face tokenizer.

        Args:
            source (str): The tokenizer source repository or path.
            revision (str | None): The model revision.
            trust_remote_code (bool): Whether to trust remote code.

        Returns:
            Any: The loaded tokenizer instance.
        """
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            source,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )


__all__ = ["TakTinyLoader"]
